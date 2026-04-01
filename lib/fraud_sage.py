"""Heterogeneous GraphSAGE model for fraud detection."""

import warnings

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch_geometric.nn.conv import HeteroConv, MessagePassing
from torch_geometric.nn.dense import Linear

warnings.filterwarnings("ignore", message=".*do not occur as destination type.*")


def focal_loss(
    y_pred: torch.Tensor,
    y_true: torch.Tensor,
    alpha: float = 0.25,
    gamma: float = 2.0,
    sample_weight: torch.Tensor | None = None,
) -> torch.Tensor:
    """Focal loss for extreme class imbalance.  Ignores nodes labelled -1."""
    mask = y_true != -1
    if mask.sum() == 0:
        return torch.tensor(0.0, requires_grad=True, device=y_pred.device)

    logits = y_pred[mask]
    targets = y_true[mask].float()
    probs = torch.sigmoid(logits)
    bce = F.binary_cross_entropy_with_logits(logits, targets, reduction="none")
    p_t = targets * probs + (1 - targets) * (1 - probs)
    loss = bce * ((1 - p_t) ** gamma)

    if alpha >= 0:
        alpha_t = targets * alpha + (1 - targets) * (1 - alpha)
        loss = alpha_t * loss

    if sample_weight is not None:
        loss = loss * sample_weight[mask]

    return loss.mean()


class MaxPoolConcatSAGEConv(MessagePassing):
    """GraphSAGE-style max-pooling aggregator with self/neighbour concatenation."""

    def __init__(self, in_channels, out_channels):
        super().__init__(aggr="max")
        if isinstance(in_channels, int):
            in_channels = (in_channels, in_channels)

        self.neigh_mlp = nn.Sequential(
            Linear(in_channels[0], out_channels),
            nn.ReLU(),
            Linear(out_channels, out_channels),
            nn.ReLU(),
        )
        self.concat_proj = Linear(2 * out_channels, out_channels)
        self.self_proj = Linear(in_channels[1], out_channels)

    def forward(self, x, edge_index, edge_weight=None, edge_attr=None):
        if isinstance(x, torch.Tensor):
            x = (x, x)
        x_src, x_dst = x

        x_src_t = self.neigh_mlp(x_src)
        x_dst_t = self.self_proj(x_dst)

        if edge_weight is None:
            eff_weight = torch.ones(
                edge_index.size(1), device=edge_index.device, dtype=x_src_t.dtype
            )
        else:
            eff_weight = edge_weight

        # Optional CNN-risk gate on edge attributes.
        if edge_attr is not None and edge_attr.dim() == 2 and edge_attr.size(1) >= 2:
            txn_prob_gate = edge_attr[:, 1].clamp(0.0, 1.0)
            eff_weight = eff_weight * (0.5 + txn_prob_gate)

        pooled = self.propagate(
            edge_index,
            x=x_src_t,
            edge_weight=eff_weight,
            size=(x_src_t.size(0), x_dst_t.size(0)),
        )
        # Replace -inf from empty neighborhoods with zeros.
        pooled = torch.where(torch.isfinite(pooled), pooled, torch.zeros_like(pooled))

        return self.concat_proj(torch.cat([x_dst_t, pooled], dim=-1))

    def message(self, x_j, *, edge_weight): # type: ignore
        return x_j * edge_weight.view(-1, 1)


class FraudSAGE(torch.nn.Module):
    """3-layer heterogeneous GraphSAGE with max-pooling aggregation.

    Layer 1: Category/City → Customer  (aggregate hub signals)
    Layer 2: Customer → Category/City  (update hub embeddings)
    Layer 3: Category/City → Customer  (final aggregation)

    A residual connection from the input projection is added to the Layer-1
    output for stable gradient flow.
    """

    def __init__(
        self,
        hidden_channels: int,
        identity_feature_cols: tuple[int, ...] = (0, 1, 2, 3, 4),
        identity_feature_dropout_p: float = 0.10,
    ):
        super().__init__()

        self.identity_feature_cols = tuple(identity_feature_cols)
        self.identity_feature_dropout_p = float(identity_feature_dropout_p)

        self.conv1 = HeteroConv(
            {
                ("category", "rev_purchases_at", "customer"): MaxPoolConcatSAGEConv(
                    (-1, -1), hidden_channels
                ),
                ("city", "rev_transacts_in", "customer"): MaxPoolConcatSAGEConv(
                    (-1, -1), hidden_channels
                ),
            },
            aggr="max",
        )
        self.bn1 = nn.BatchNorm1d(hidden_channels)

        self.conv2 = HeteroConv(
            {
                ("customer", "purchases_at", "category"): MaxPoolConcatSAGEConv(
                    (-1, -1), hidden_channels
                ),
                ("customer", "transacts_in", "city"): MaxPoolConcatSAGEConv(
                    (-1, -1), hidden_channels
                ),
            },
            aggr="max",
        )

        self.conv3 = HeteroConv(
            {
                ("category", "rev_purchases_at", "customer"): MaxPoolConcatSAGEConv(
                    (-1, -1), hidden_channels
                ),
                ("city", "rev_transacts_in", "customer"): MaxPoolConcatSAGEConv(
                    (-1, -1), hidden_channels
                ),
            },
            aggr="max",
        )
        self.bn2 = nn.BatchNorm1d(hidden_channels)

        self.input_proj = Linear(-1, hidden_channels)
        self.classifier = nn.Sequential(
            nn.Linear(hidden_channels, hidden_channels // 2),
            nn.ELU(),
            nn.Dropout(p=0.2),
            nn.Linear(hidden_channels // 2, 1),
        )

    def encode(self, x_dict, edge_index_dict, edge_weight_dict=None, edge_attr_dict=None):
        if edge_weight_dict is None:
            edge_weight_dict = {
                et: torch.ones(ei.size(1), device=ei.device)
                for et, ei in edge_index_dict.items()
            }
        if edge_attr_dict is None:
            edge_attr_dict = {}

        x_customer = x_dict["customer"]
        if self.training and self.identity_feature_dropout_p > 0.0 and x_customer.size(1) > 0:
            valid_cols = [c for c in self.identity_feature_cols if 0 <= c < x_customer.size(1)]
            if valid_cols:
                x_customer = x_customer.clone()
                col_idx = torch.tensor(valid_cols, dtype=torch.long, device=x_customer.device)
                x_customer[:, col_idx] = F.dropout(
                    x_customer[:, col_idx],
                    p=self.identity_feature_dropout_p,
                    training=True,
                )

        x_work = dict(x_dict)
        x_work["customer"] = x_customer

        res = F.elu(self.input_proj(x_work["customer"]))

        h1 = self.conv1(
            x_work,
            edge_index_dict,
            edge_weight_dict=edge_weight_dict,
            edge_attr_dict=edge_attr_dict,
        )
        h1["customer"] = F.elu(self.bn1(h1["customer"]) + res)
        for k in x_work:
            if k not in h1:
                h1[k] = x_work[k]

        h2 = self.conv2(
            h1,
            edge_index_dict,
            edge_weight_dict=edge_weight_dict,
            edge_attr_dict=edge_attr_dict,
        )
        h2 = {k: F.elu(v) for k, v in h2.items()}
        if "customer" not in h2:
            h2["customer"] = h1["customer"]

        h3 = self.conv3(
            h2,
            edge_index_dict,
            edge_weight_dict=edge_weight_dict,
            edge_attr_dict=edge_attr_dict,
        )
        h3["customer"] = F.elu(self.bn2(h3["customer"]) + h1["customer"])

        return h3["customer"]

    def forward(self, x_dict, edge_index_dict, edge_weight_dict=None, edge_attr_dict=None):
        customer_embeddings = self.encode(
            x_dict,
            edge_index_dict,
            edge_weight_dict=edge_weight_dict,
            edge_attr_dict=edge_attr_dict,
        )

        return self.classifier(customer_embeddings)
