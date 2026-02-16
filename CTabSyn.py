import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader, TensorDataset
import pandas as pd
import numpy as np
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import StratifiedKFold
from sklearn.preprocessing import StandardScaler
from tqdm import tqdm

# --- 1. ORD Labeling Logic ---
def get_ord_labels(X, y, k=5, threshold=0.35):
    """
    Ternary Labeling:
    0: Clear Majority (Normal)
    1: Minority (Fraud)
    2: Overlap Region (High-risk Normal)
    """
    ord_labels = np.zeros(len(y))
    ord_labels[y == 1] = 1 # Mark actual fraud
    
    skf = StratifiedKFold(n_splits=k)
    for train_idx, val_idx in tqdm(skf.split(X, y), desc="Getting ORD Labels"):
        rf = RandomForestClassifier(n_estimators=100, class_weight='balanced', n_jobs=-1)
        rf.fit(X.iloc[train_idx], y.iloc[train_idx])
        
        # Predict probability of fraud for legitimate cases
        probs = rf.predict_proba(X.iloc[val_idx])[:, 1]
        
        val_y = y.iloc[val_idx].values
        # Legitimate transactions that 'look' like fraud go to Overlap (Class 2)
        overlap_mask = (val_y == 0) & (probs >= threshold)
        
        actual_indices = X.index[val_idx]
        ord_labels[actual_indices[overlap_mask]] = 2
        
    return ord_labels

# --- 2. Conditional VAE (CTabSyn Core) ---
class TabularVAE(nn.Module):
    def __init__(self, input_dim, cond_dim=3, latent_dim=32):
        super(TabularVAE, self).__init__()
        
        # Encoder: Feature + Ternary Label -> Latent Space
        self.encoder = nn.Sequential(
            nn.Linear(input_dim + cond_dim, 128),
            nn.BatchNorm1d(128),
            nn.ReLU(),
            nn.Linear(128, 64),
            nn.ReLU()
        )
        self.fc_mu = nn.Linear(64, latent_dim)
        self.fc_logvar = nn.Linear(64, latent_dim)
        
        # Decoder: Latent Space + Ternary Label -> Feature Reconstruction
        self.decoder = nn.Sequential(
            nn.Linear(latent_dim + cond_dim, 64),
            nn.ReLU(),
            nn.Linear(64, 128),
            nn.BatchNorm1d(128),
            nn.ReLU(),
            nn.Linear(128, input_dim),
            nn.Sigmoid() # Assuming data is scaled [0, 1]
        )

    def reparameterize(self, mu, logvar):
        std = torch.exp(0.5 * logvar)
        eps = torch.randn_like(std)
        return mu + eps * std

    def forward(self, x, labels):
        # One-hot encode labels (0, 1, 2)
        cond = F.one_hot(labels.long(), num_classes=3).float()
        
        # Encode
        enc_input = torch.cat([x, cond], dim=1)
        h = self.encoder(enc_input)
        mu, logvar = self.fc_mu(h), self.fc_logvar(h)
        
        # Reparameterize
        z = self.reparameterize(mu, logvar)
        
        # Decode
        dec_input = torch.cat([z, cond], dim=1)
        recon_x = self.decoder(dec_input)
        
        return recon_x, mu, logvar

# --- 3. Training & Generation ---
def train_and_generate(df, target_col='fraud', target_fraud_ratio=0.10):
    # Prepare Data
    X = df.drop(columns=[target_col])
    y = df[target_col]
    
    # Scale features for VAE [0, 1]
    scaler = StandardScaler()
    X_scaled = (X - X.min()) / (X.max() - X.min() + 1e-7) # Simple min-max
    
    # 1. Get ORD Labels
    print("Performing Overlap Region Detection...")
    ord_labels = get_ord_labels(X, y)
    
    # Convert to Tensors
    X_tensor = torch.FloatTensor(X_scaled.values)
    L_tensor = torch.LongTensor(ord_labels)
    dataset = TensorDataset(X_tensor, L_tensor)
    loader = DataLoader(dataset, batch_size=256, shuffle=True)
    
    # 2. Train VAE
    model = TabularVAE(input_dim=X.shape[1])
    optimizer = torch.optim.Adam(model.parameters(), lr=1e-3)
    
    print("Training VAE...")
    model.train()
    for epoch in tqdm(range(50),desc="Training CTabSyn"): # 50 epochs is usually enough for tabular
        for batch_x, batch_l in loader:
            recon, mu, logvar = model(batch_x, batch_l)
            
            # Loss = Reconstruction (MSE) + KL Divergence
            recon_loss = F.mse_loss(recon, batch_x, reduction='sum')
            kl_loss = -0.5 * torch.sum(1 + logvar - mu.pow(2) - logvar.exp())
            
            loss = recon_loss + kl_loss
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

    # 3. Generate Synthetic Fraud (Class 1)
    # Calculate how many we need to hit 10%
    total_samples = len(df)
    current_fraud = (y == 1).sum()
    needed_fraud = int(total_samples * target_fraud_ratio) - current_fraud
    
    print(f"Generating {needed_fraud} synthetic fraud samples...")
    model.eval()
    with torch.no_grad():
        z = torch.randn(needed_fraud, 32)
        cond = torch.zeros(needed_fraud).long() + 1 # Target label 1 (Fraud)
        cond_one_hot = F.one_hot(cond, num_classes=3).float()
        
        gen_input = torch.cat([z, cond_one_hot], dim=1)
        synth_scaled = model.decoder(gen_input).numpy()
        
    # Inverse Scale
    synth_df = pd.DataFrame(synth_scaled, columns=X.columns)
    synth_df = synth_df * (X.max() - X.min() + 1e-7) + X.min()
    synth_df[target_col] = 1
    
    return pd.concat([df, synth_df], axis=0).sample(frac=1).reset_index(drop=True)