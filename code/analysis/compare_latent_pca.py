import os
from dataclasses import dataclass

import matplotlib.pyplot as plt
import pandas as pd
import torch
import torch.nn as nn
import torch.nn.functional as F
from sklearn.decomposition import PCA
from torch.utils.data import Dataset, DataLoader

# Config

@dataclass
class Config:
    csv_path: str = "CoV-AbDab.csv"
    antigen_col: str = "antigen"
    cdr3_col: str = "cdr3"

    max_cdr3_len: int = 30
    batch_size: int = 256

    finetune_ckpt: str = "conditional_cvae_finetune.pt"
    scratch_ckpt: str = "conditional_cvae_scratch.pt"

    device: str = "cuda" if torch.cuda.is_available() else "cpu"

    mu_csv: str = "latent_mu_q_comparison.csv"
    z_csv: str = "latent_z_cond_comparison.csv"
    mu_plot: str = "pca_mu_q_comparison.png"
    z_plot: str = "pca_z_cond_comparison.png"


cfg = Config()

# Vocabulary

AMINO_ACIDS = list("ACDEFGHIKLMNPQRSTVWY")
SPECIAL_TOKENS = ["X", "<PAD>"]

itos = AMINO_ACIDS + SPECIAL_TOKENS
stoi = {ch: i for i, ch in enumerate(itos)}

PAD_IDX = stoi["<PAD>"]
UNK_IDX = stoi["X"]
VOCAB_SIZE = len(itos)

# Dataset

class AntigenCDR3Dataset(Dataset):
    def __init__(self, csv_path, antigen_col, cdr3_col, max_cdr3_len):
        self.df = pd.read_csv(csv_path)

        if antigen_col not in self.df.columns:
            raise ValueError(f"Column '{antigen_col}' not found in {csv_path}")
        if cdr3_col not in self.df.columns:
            raise ValueError(f"Column '{cdr3_col}' not found in {csv_path}")

        df = self.df[[antigen_col, cdr3_col]].dropna().copy()
        df[antigen_col] = df[antigen_col].astype(str).str.strip().str.upper()
        df[cdr3_col] = df[cdr3_col].astype(str).str.strip().str.upper()

        df = df[(df[antigen_col].str.len() > 0) & (df[cdr3_col].str.len() > 0)].reset_index(drop=True)
        if len(df) == 0:
            raise ValueError("No valid rows after filtering.")

        self.samples = list(zip(df[antigen_col].tolist(), df[cdr3_col].tolist()))
        self.max_antigen_len = max(len(a) for a, _ in self.samples)
        self.max_cdr3_len = max_cdr3_len

        print(f"Loaded {len(self.samples)} paired samples")
        print(f"Max antigen length = {self.max_antigen_len}")
        print(f"Max cdr3 length cap = {self.max_cdr3_len}")

    def __len__(self):
        return len(self.samples)

    def encode_seq(self, seq, fixed_len):
        true_len = min(len(seq), fixed_len)
        seq = seq[:fixed_len]
        ids = [stoi.get(ch, UNK_IDX) for ch in seq]
        if len(ids) < fixed_len:
            ids += [PAD_IDX] * (fixed_len - len(ids))
        return torch.tensor(ids, dtype=torch.long), torch.tensor(true_len, dtype=torch.long)

    def __getitem__(self, idx):
        antigen, cdr3 = self.samples[idx]
        a, a_len = self.encode_seq(antigen, self.max_antigen_len)
        x, x_len = self.encode_seq(cdr3, self.max_cdr3_len)
        a_mask = (a != PAD_IDX).long()

        return {
            "index": idx,
            "antigen": antigen,
            "cdr3": cdr3,
            "x": x,
            "x_len": x_len,
            "a": a,
            "a_mask": a_mask,
            "a_len": a_len,
        }

# Model blocks

class ResBlock1D(nn.Module):
    def __init__(self, channels: int, kernel_size: int = 3, dropout: float = 0.1):
        super().__init__()
        padding = kernel_size // 2
        self.conv1 = nn.Conv1d(channels, channels, kernel_size, padding=padding)
        self.bn1 = nn.BatchNorm1d(channels)
        self.conv2 = nn.Conv1d(channels, channels, kernel_size, padding=padding)
        self.bn2 = nn.BatchNorm1d(channels)
        self.dropout = nn.Dropout(dropout)

    def forward(self, x):
        residual = x
        out = self.conv1(x)
        out = self.bn1(out)
        out = F.relu(out, inplace=True)
        out = self.dropout(out)

        out = self.conv2(out)
        out = self.bn2(out)
        out = out + residual
        out = F.relu(out, inplace=True)
        return out


class CNNEncoder(nn.Module):
    def __init__(
        self,
        vocab_size: int,
        embed_dim: int,
        hidden_dim: int,
        latent_dim: int,
        max_seq_len: int,
        num_layers: int = 5,
        kernel_size: int = 3,
        dropout: float = 0.1,
    ):
        super().__init__()
        self.embedding = nn.Embedding(vocab_size, embed_dim, padding_idx=PAD_IDX)
        self.input_proj = nn.Conv1d(embed_dim, hidden_dim, kernel_size=1)
        self.blocks = nn.ModuleList(
            [ResBlock1D(hidden_dim, kernel_size, dropout) for _ in range(num_layers)]
        )

        flat_dim = hidden_dim * max_seq_len
        self.fc_mu = nn.Linear(flat_dim, latent_dim)
        self.fc_logvar = nn.Linear(flat_dim, latent_dim)

        self.max_seq_len = max_seq_len
        self.hidden_dim = hidden_dim

    def encode_feature(self, x):
        emb = self.embedding(x)
        emb = emb.transpose(1, 2)
        h = self.input_proj(emb)
        for block in self.blocks:
            h = block(h)
        h = h.reshape(h.size(0), -1)
        return h

    def forward(self, x):
        h = self.encode_feature(x)
        mu = self.fc_mu(h)
        logvar = self.fc_logvar(h)
        return mu, logvar


class CNNDecoder(nn.Module):
    def __init__(
        self,
        latent_dim: int,
        hidden_dim: int,
        vocab_size: int,
        max_seq_len: int,
        num_layers: int = 5,
        kernel_size: int = 3,
        dropout: float = 0.1,
    ):
        super().__init__()
        self.max_seq_len = max_seq_len
        self.hidden_dim = hidden_dim

        self.fc = nn.Linear(latent_dim, hidden_dim * max_seq_len)
        self.blocks = nn.ModuleList(
            [ResBlock1D(hidden_dim, kernel_size, dropout) for _ in range(num_layers)]
        )
        self.output_proj = nn.Conv1d(hidden_dim, vocab_size, kernel_size=1)

    def forward(self, z):
        h = self.fc(z)
        h = h.view(z.size(0), self.hidden_dim, self.max_seq_len)
        for block in self.blocks:
            h = block(h)
        logits = self.output_proj(h)
        logits = logits.transpose(1, 2)
        return logits


class AntigenEncoder(nn.Module):
    def __init__(
        self,
        vocab_size: int,
        embed_dim: int,
        hidden_dim: int,
        num_layers: int = 3,
        kernel_size: int = 3,
        dropout: float = 0.1,
    ):
        super().__init__()
        self.embedding = nn.Embedding(vocab_size, embed_dim, padding_idx=PAD_IDX)
        self.input_proj = nn.Conv1d(embed_dim, hidden_dim, kernel_size=1)
        self.blocks = nn.ModuleList(
            [ResBlock1D(hidden_dim, kernel_size, dropout) for _ in range(num_layers)]
        )

    def forward(self, a, a_mask):
        emb = self.embedding(a)
        emb = emb.transpose(1, 2)
        h = self.input_proj(emb)
        for block in self.blocks:
            h = block(h)

        mask = a_mask.unsqueeze(1).float()
        h = h * mask
        pooled = h.sum(dim=2) / mask.sum(dim=2).clamp_min(1.0)
        return pooled


class ConditionalCNNVAE(nn.Module):
    def __init__(self, config: dict):
        super().__init__()

        self.encoder = CNNEncoder(
            vocab_size=VOCAB_SIZE,
            embed_dim=config["embed_dim"],
            hidden_dim=config["hidden_dim"],
            latent_dim=config["latent_dim"],
            max_seq_len=config["max_cdr3_len"],
            num_layers=config["num_layers"],
            kernel_size=config["kernel_size"],
            dropout=config["dropout"],
        )

        self.decoder = CNNDecoder(
            latent_dim=config["latent_dim"],
            hidden_dim=config["hidden_dim"],
            vocab_size=VOCAB_SIZE,
            max_seq_len=config["max_cdr3_len"],
            num_layers=config["num_layers"],
            kernel_size=config["kernel_size"],
            dropout=config["dropout"],
        )

        self.antigen_encoder = AntigenEncoder(
            vocab_size=VOCAB_SIZE,
            embed_dim=config["embed_dim"],
            hidden_dim=config["antigen_hidden_dim"],
            num_layers=config["antigen_num_layers"],
            kernel_size=config["kernel_size"],
            dropout=config["dropout"],
        )

        seq_feat_dim = config["hidden_dim"] * config["max_cdr3_len"]
        ant_feat_dim = config["antigen_hidden_dim"]

        self.posterior_mu = nn.Sequential(
            nn.Linear(seq_feat_dim + ant_feat_dim, config["fusion_dim"]),
            nn.ReLU(),
            nn.Dropout(config["dropout"]),
            nn.Linear(config["fusion_dim"], config["latent_dim"]),
        )
        self.posterior_logvar = nn.Sequential(
            nn.Linear(seq_feat_dim + ant_feat_dim, config["fusion_dim"]),
            nn.ReLU(),
            nn.Dropout(config["dropout"]),
            nn.Linear(config["fusion_dim"], config["latent_dim"]),
        )

        self.prior_mu = nn.Sequential(
            nn.Linear(ant_feat_dim, config["fusion_dim"]),
            nn.ReLU(),
            nn.Dropout(config["dropout"]),
            nn.Linear(config["fusion_dim"], config["latent_dim"]),
        )
        self.prior_logvar = nn.Sequential(
            nn.Linear(ant_feat_dim, config["fusion_dim"]),
            nn.ReLU(),
            nn.Dropout(config["dropout"]),
            nn.Linear(config["fusion_dim"], config["latent_dim"]),
        )

        self.decoder_input = nn.Sequential(
            nn.Linear(config["latent_dim"] + ant_feat_dim, config["fusion_dim"]),
            nn.ReLU(),
            nn.Dropout(config["dropout"]),
            nn.Linear(config["fusion_dim"], config["latent_dim"]),
        )

        self.length_head = nn.Sequential(
            nn.Linear(config["latent_dim"], config["latent_dim"]),
            nn.ReLU(),
            nn.Dropout(config["dropout"]),
            nn.Linear(config["latent_dim"], config["max_cdr3_len"]),
        )

    @staticmethod
    def reparameterize(mu, logvar):
        std = torch.exp(0.5 * logvar)
        eps = torch.randn_like(std)
        return mu + eps * std

# Feature extraction

@torch.no_grad()
def extract_latents(model, loader, device, model_name):
    model.eval()

    mu_rows = []
    z_rows = []

    for batch in loader:
        x = batch["x"].to(device)
        a = batch["a"].to(device)
        a_mask = batch["a_mask"].to(device)

        idxs = batch["index"].tolist()
        antigens = batch["antigen"]
        cdr3s = batch["cdr3"]
        x_lens = batch["x_len"].tolist()
        a_lens = batch["a_len"].tolist()

        hx = model.encoder.encode_feature(x)
        ha = model.antigen_encoder(a, a_mask)

        q_input = torch.cat([hx, ha], dim=-1)
        mu_q = model.posterior_mu(q_input)
        logvar_q = model.posterior_logvar(q_input)
        z = model.reparameterize(mu_q, logvar_q)

        z_cond = model.decoder_input(torch.cat([z, ha], dim=-1))

        mu_q_np = mu_q.cpu().numpy()
        z_cond_np = z_cond.cpu().numpy()

        for i in range(len(idxs)):
            base_info = {
                "sample_index": idxs[i],
                "model": model_name,
                "cdr3": cdr3s[i],
                "cdr3_len": int(x_lens[i]),
                "antigen_len": int(a_lens[i]),
                "antigen": antigens[i],
            }

            mu_row = dict(base_info)
            for j, v in enumerate(mu_q_np[i]):
                mu_row[f"dim_{j}"] = float(v)
            mu_rows.append(mu_row)

            z_row = dict(base_info)
            for j, v in enumerate(z_cond_np[i]):
                z_row[f"dim_{j}"] = float(v)
            z_rows.append(z_row)

    return pd.DataFrame(mu_rows), pd.DataFrame(z_rows)

# PCA plotting

def make_pca_plot(df, output_png, title):
    feature_cols = [c for c in df.columns if c.startswith("dim_")]
    X = df[feature_cols].values

    pca = PCA(n_components=2)
    pcs = pca.fit_transform(X)

    plot_df = df.copy()
    plot_df["PC1"] = pcs[:, 0]
    plot_df["PC2"] = pcs[:, 1]

    plt.figure(figsize=(8, 6))
    for model_name in sorted(plot_df["model"].unique()):
        sub = plot_df[plot_df["model"] == model_name]
        plt.scatter(sub["PC1"], sub["PC2"], s=8, alpha=0.5, label=model_name)

    plt.xlabel(f"PC1 ({pca.explained_variance_ratio_[0] * 100:.1f}%)")
    plt.ylabel(f"PC2 ({pca.explained_variance_ratio_[1] * 100:.1f}%)")
    plt.title(title)
    plt.legend()
    plt.tight_layout()
    plt.savefig(output_png, dpi=300)
    plt.close()

    return pca.explained_variance_ratio_

# Main

def main():
    dataset = AntigenCDR3Dataset(
        cfg.csv_path,
        cfg.antigen_col,
        cfg.cdr3_col,
        cfg.max_cdr3_len,
    )
    loader = DataLoader(dataset, batch_size=cfg.batch_size, shuffle=False, drop_last=False)

    results_mu = []
    results_z = []

    checkpoints = [
        ("finetune", cfg.finetune_ckpt),
        ("scratch", cfg.scratch_ckpt),
    ]

    for model_name, ckpt_path in checkpoints:
        if not os.path.exists(ckpt_path):
            print(f"Skip {model_name}: checkpoint not found -> {ckpt_path}")
            continue

        print(f"Loading {model_name} from {ckpt_path}")
        ckpt = torch.load(ckpt_path, map_location=cfg.device)
        model = ConditionalCNNVAE(ckpt["config"]).to(cfg.device)
        model.load_state_dict(ckpt["model_state_dict"], strict=True)

        mu_df, z_df = extract_latents(model, loader, cfg.device, model_name)
        results_mu.append(mu_df)
        results_z.append(z_df)

    if not results_mu or not results_z:
        raise RuntimeError("No checkpoint could be loaded.")

    mu_all = pd.concat(results_mu, ignore_index=True)
    z_all = pd.concat(results_z, ignore_index=True)

    mu_all.to_csv(cfg.mu_csv, index=False)
    z_all.to_csv(cfg.z_csv, index=False)

    mu_var = make_pca_plot(mu_all, cfg.mu_plot, "PCA of posterior mean (mu_q)")
    z_var = make_pca_plot(z_all, cfg.z_plot, "PCA of conditional latent (z_cond)")

    print(f"Saved: {cfg.mu_csv}")
    print(f"Saved: {cfg.z_csv}")
    print(f"Saved: {cfg.mu_plot}")
    print(f"Saved: {cfg.z_plot}")
    print(f"mu_q PCA variance ratio: {mu_var}")
    print(f"z_cond PCA variance ratio: {z_var}")


if __name__ == "__main__":
    main()

