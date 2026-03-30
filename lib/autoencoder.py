"""Autoencoder for unsupervised transaction anomaly detection."""

from __future__ import annotations

import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from sklearn.preprocessing import StandardScaler
from torch.utils.data import DataLoader, TensorDataset, WeightedRandomSampler


class TransactionAutoencoder(nn.Module):
    """Autoencoder for anomaly detection on transaction-level behavioral features."""

    def __init__(self, input_dim: int, encoding_dim: int = 16):
        super().__init__()
        self.encoder = nn.Sequential(
            nn.Linear(input_dim, 64),
            nn.ReLU(),
            nn.Linear(64, 32),
            nn.ReLU(),
            nn.Linear(32, encoding_dim),
        )
        self.decoder = nn.Sequential(
            nn.Linear(encoding_dim, 32),
            nn.ReLU(),
            nn.Linear(32, 64),
            nn.ReLU(),
            nn.Linear(64, input_dim),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.decoder(self.encoder(x))

    def encode(self, x: torch.Tensor) -> torch.Tensor:
        return self.encoder(x)


def train_autoencoder(
    X_train: np.ndarray,
    epochs: int = 50,
    batch_size: int = 128,
    encoding_dim: int = 16,
) -> tuple[TransactionAutoencoder, StandardScaler]:
    """Fit a StandardScaler, then train the autoencoder on all transactions.

    Returns the trained model (eval mode) and the fitted scaler so that the
    same transform can be applied when scoring new data.
    """
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X_train)
    X_scaled = np.nan_to_num(X_scaled, nan=0.0, posinf=3.0, neginf=-3.0).astype(np.float32)

    ds = TensorDataset(torch.tensor(X_scaled, dtype=torch.float32))
    dl = DataLoader(ds, batch_size=batch_size, shuffle=True)

    model = TransactionAutoencoder(input_dim=X_scaled.shape[1], encoding_dim=encoding_dim)
    model = model.to(device)
    criterion = nn.MSELoss()
    optimizer = optim.Adam(model.parameters(), lr=1e-3, weight_decay=1e-5)

    print(f"Training autoencoder on {len(X_scaled):,} transactions...")
    model.train()
    for epoch in range(epochs):
        epoch_loss = 0.0
        for (xb,) in dl:
            xb = xb.to(device)
            optimizer.zero_grad()
            loss = criterion(model(xb), xb)
            loss.backward()
            optimizer.step()
            epoch_loss += float(loss.item())
        if (epoch + 1) % 10 == 0:
            print(f"  Epoch {epoch + 1}/{epochs}, Loss: {epoch_loss / len(dl):.6f}")

    model.eval()
    return model, scaler


def fine_tune_autoencoder(
    model: TransactionAutoencoder,
    scaler: StandardScaler,
    X_train: np.ndarray,
    labels: np.ndarray,
    epochs: int = 30,
    batch_size: int = 128,
    learning_rate: float = 5e-4,
) -> TransactionAutoencoder:
    """Fine-tune an autoencoder on labeled data with inverse-frequency sampling."""
    device = next(model.parameters()).device

    X_scaled = scaler.transform(X_train)
    X_scaled = np.nan_to_num(X_scaled, nan=0.0, posinf=3.0, neginf=-3.0).astype(np.float32)
    y = np.asarray(labels).astype(np.int64)

    class_counts = np.bincount(y, minlength=2).astype(np.float64)
    class_weights = np.ones_like(class_counts, dtype=np.float64)
    nonzero = class_counts > 0
    class_weights[nonzero] = class_counts[nonzero].sum() / class_counts[nonzero]
    sample_weights: np.ndarray = class_weights[y]
    weights_list: list[float] = sample_weights.astype(float).tolist()

    ds = TensorDataset(
        torch.tensor(X_scaled, dtype=torch.float32),
        torch.tensor(y, dtype=torch.long),
    )
    sampler = WeightedRandomSampler(
        weights=weights_list,
        num_samples=len(sample_weights),
        replacement=True,
    )
    dl = DataLoader(ds, batch_size=batch_size, sampler=sampler)

    criterion = nn.MSELoss(reduction="none")
    optimizer = optim.Adam(model.parameters(), lr=learning_rate, weight_decay=1e-5)

    print(f"Fine-tuning autoencoder on {len(X_scaled):,} labeled transactions...")
    model.train()
    for epoch in range(epochs):
        epoch_loss = 0.0
        for xb, yb in dl:
            xb = xb.to(device)
            yb = yb.to(device)
            optimizer.zero_grad()

            recon = model(xb)
            per_row_loss = criterion(recon, xb).mean(dim=1)

            # Slightly up-weight fraud-customer transactions during fine-tuning.
            row_weight = torch.where(yb == 1, torch.tensor(1.25, device=device), torch.tensor(1.0, device=device))
            loss = (per_row_loss * row_weight).mean()
            loss.backward()
            optimizer.step()
            epoch_loss += float(loss.item())

        if (epoch + 1) % 10 == 0 or epoch == 0:
            print(f"  Fine-tune epoch {epoch + 1}/{epochs}, Loss: {epoch_loss / len(dl):.6f}")

    model.eval()
    return model


def detect_anomalies(
    model: TransactionAutoencoder,
    scaler: StandardScaler,
    X: np.ndarray,
    threshold_percentile: int = 95,
) -> tuple[np.ndarray, np.ndarray, float]:
    """Score every row with per-sample MSE reconstruction error.

    Returns:
        anomalies: boolean mask, True where error > threshold
        recon_errors: float32 array of per-row MSE
        threshold: the percentile cutoff used
    """
    device = next(model.parameters()).device

    X_scaled = scaler.transform(X)
    X_scaled = np.nan_to_num(X_scaled, nan=0.0, posinf=3.0, neginf=-3.0).astype(np.float32)

    recon_errors = np.zeros(len(X_scaled), dtype=np.float32)
    model.eval()
    with torch.no_grad():
        batch_size = 1024
        for i in range(0, len(X_scaled), batch_size):
            xb = torch.tensor(X_scaled[i : i + batch_size], dtype=torch.float32, device=device)
            recon = model(xb)
            errors = torch.mean((xb - recon) ** 2, dim=1)
            recon_errors[i : i + batch_size] = errors.cpu().numpy()

    threshold = float(np.percentile(recon_errors, threshold_percentile))
    anomalies = recon_errors > threshold
    return anomalies, recon_errors, threshold


def save_autoencoder(
    model: TransactionAutoencoder,
    scaler: StandardScaler,
    checkpoint_path: str,
    *,
    threshold: float | None = None,
    threshold_percentile: int | None = None,
    feature_names: list[str] | None = None,
    metadata: dict | None = None,
) -> None:
    """Persist the trained autoencoder, scaler, and scoring metadata."""
    encoder_last = model.encoder[-1]
    checkpoint = {
        "state_dict": model.state_dict(),
        "input_dim": int(model.encoder[0].in_features),
        "encoding_dim": int(encoder_last.out_features),
        "scaler": scaler,
        "threshold": threshold,
        "threshold_percentile": threshold_percentile,
        "feature_names": feature_names,
        "metadata": metadata or {},
    }
    torch.save(checkpoint, checkpoint_path)


def load_autoencoder(
    checkpoint_path: str,
    *,
    map_location: str | torch.device | None = None,
) -> tuple[TransactionAutoencoder, StandardScaler, dict]:
    """Load a persisted autoencoder checkpoint and return model, scaler, metadata."""
    checkpoint = torch.load(checkpoint_path, map_location=map_location or "cpu", weights_only=False)
    model = TransactionAutoencoder(
        input_dim=int(checkpoint["input_dim"]),
        encoding_dim=int(checkpoint["encoding_dim"]),
    )
    model.load_state_dict(checkpoint["state_dict"])
    model.eval()

    scaler = checkpoint["scaler"]
    metadata = {
        "threshold": checkpoint.get("threshold"),
        "threshold_percentile": checkpoint.get("threshold_percentile"),
        "feature_names": checkpoint.get("feature_names"),
    }
    metadata.update(checkpoint.get("metadata", {}))
    return model, scaler, metadata
