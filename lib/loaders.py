"""NeighborLoader factory for the fraud detection heterogeneous graph."""

from torch_geometric.data import HeteroData
from torch_geometric.loader import NeighborLoader


def build_neighbor_loaders(
    data: HeteroData,
    num_neighbors: list[int],
    batch_size: int,
) -> tuple[NeighborLoader, NeighborLoader]:
    """Return (train_loader, val_loader) sampling from *data*.

    Args:
        data: The heterogeneous graph with ``train_mask`` and ``val_mask`` set
              on the ``'customer'`` node type.
        num_neighbors: Per-hop neighbour counts, e.g. ``[15, 10]`` for 2-hop.
        batch_size: Number of seed nodes per mini-batch.
    """
    train_loader = NeighborLoader(
        data,
        num_neighbors=num_neighbors,
        batch_size=batch_size,
        input_nodes=("customer", data["customer"].train_mask),
        shuffle=True,
        num_workers=0,
    )
    val_loader = NeighborLoader(
        data,
        num_neighbors=num_neighbors,
        batch_size=batch_size,
        input_nodes=("customer", data["customer"].val_mask),
        shuffle=False,
        num_workers=0,
    )
    return train_loader, val_loader
