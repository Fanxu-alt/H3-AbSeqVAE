import os
from dataclasses import dataclass

import pandas as pd
import torch
import torch.nn as nn
import torch.nn.functional as F
import matplotlib.pyplot as plt
from sklearn.decomposition import PCA

# Config

TARGET_ANTIGEN = (
    "RVQPTESIVRFPNITNLCPFGEVFNATRFASVYAWNRKRISNCVADYSVLYNSASFSTFKCYGVSPTKLNDLCFTNVYADSFVIRGDEVRQIAPGQTGKIADYNYKLPDDFTGCVIAWNSNNLDSKVGGNYNYLYRLFRKSNLKPFERDISTEIYQAGSTPCNGVEGFNCYFPLQSYGFQPTNGVGYQPYRVVVLSFELLHAPATVCGPKKSTNLVKNKCVNF"
)

CSV_PATH = "CoV-AbDab.csv"
ANTIGEN_COL = "antigen"
CDR3_COL = "cdr3"

FINETUNE_CKPT = "conditional_cvae_finetune.pt"
SCRATCH_CKPT = "conditional_cvae_scratch.pt"

NUM_PRIOR_SAMPLES = 500
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"

OUT_CSV = "prior_posterior_projection.csv"
OUT_PNG = "prior_posterior_projection.png"

# Vocabulary

AMINO_ACIDS = list("ACDEFGHIKLMNPQRSTVWY")
SPECIAL_TOKENS = ["X", "<PAD>"]

itos = AMINO_ACIDS + SPECIAL_TOKENS
stoi = {ch: i for i, ch in enumerate(itos)}

PAD_IDX = stoi["<PAD>"]
UNK_IDX = stoi["X"]
VOCAB_SIZE = len(itos)

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
        self.hidden_dim = hidden_dim
        self.max_seq_len = max_seq_len

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

# Helpers

def encode_seq(seq, fixed_len):
    seq = str(seq).strip().upper()[:fixed_len]
    ids = [stoi.get(ch, UNK_IDX) for ch in seq]
    if len(ids) < fixed_len:
        ids += [PAD_IDX] * (fixed_len - len(ids))
    return ids


def load_model(ckpt_path):
    ckpt = torch.load(ckpt_path, map_location=DEVICE)
    config = ckpt["config"]
    model = ConditionalCNNVAE(config).to(DEVICE)
    model.load_state_dict(ckpt["model_state_dict"], strict=True)
    model.eval()
    max_antigen_len = ckpt["max_antigen_len"]
    return model, config, max_antigen_len


def get_target_rows(csv_path, antigen_col, cdr3_col, target_antigen):
    df = pd.read_csv(csv_path)
    df = df[[antigen_col, cdr3_col]].dropna().copy()
    df[antigen_col] = df[antigen_col].astype(str).str.strip().str.upper()
    df[cdr3_col] = df[cdr3_col].astype(str).str.strip().str.upper()
    target = target_antigen.strip().upper()
    sub = df[df[antigen_col] == target].reset_index(drop=True)
    return sub


@torch.no_grad()
def extract_posterior_means(model, config, max_antigen_len, df_sub):
    cdr3_max_len = config["max_cdr3_len"]

    all_mu = []
    for _, row in df_sub.iterrows():
        antigen = row[ANTIGEN_COL]
        cdr3 = row[CDR3_COL]

        x = torch.tensor(
            [encode_seq(cdr3, cdr3_max_len)], dtype=torch.long, device=DEVICE
        )
        a = torch.tensor(
            [encode_seq(antigen, max_antigen_len)], dtype=torch.long, device=DEVICE
        )
        a_mask = (a != PAD_IDX).long()

        hx = model.encoder.encode_feature(x)
        ha = model.antigen_encoder(a, a_mask)
        q_input = torch.cat([hx, ha], dim=-1)
        mu_q = model.posterior_mu(q_input)  # [1, latent_dim]
        all_mu.append(mu_q.squeeze(0).cpu())

    return torch.stack(all_mu, dim=0).numpy()


@torch.no_grad()
def sample_prior(model, max_antigen_len, target_antigen, num_samples):
    antigen = target_antigen.strip().upper()
    a = torch.tensor(
        [encode_seq(antigen, max_antigen_len)], dtype=torch.long, device=DEVICE
    )
    a_mask = (a != PAD_IDX).long()

    ha = model.antigen_encoder(a, a_mask)           # [1, H]
    ha = ha.repeat(num_samples, 1)                  # [N, H]

    mu_p = model.prior_mu(ha)
    logvar_p = model.prior_logvar(ha)
    z = model.reparameterize(mu_p, logvar_p)        # [N, latent_dim]
    return z.cpu().numpy()


def build_projection_dataframe(
    posterior_finetune,
    prior_finetune,
    posterior_scratch,
    prior_scratch,
):
    X = []
    meta = []

    def add(arr, model_name, source_name):
        for row in arr:
            X.append(row)
            meta.append((model_name, source_name))

    add(posterior_finetune, "finetune", "posterior")
    add(prior_finetune, "finetune", "prior")
    add(posterior_scratch, "scratch", "posterior")
    add(prior_scratch, "scratch", "prior")

    X = torch.tensor(X, dtype=torch.float32).numpy()
    pca = PCA(n_components=2, random_state=42)
    Z2 = pca.fit_transform(X)

    records = []
    for i, (model_name, source_name) in enumerate(meta):
        records.append({
            "model": model_name,
            "source": source_name,
            "pc1": float(Z2[i, 0]),
            "pc2": float(Z2[i, 1]),
        })

    df_plot = pd.DataFrame(records)
    return df_plot, pca.explained_variance_ratio_


def plot_projection(df_plot, explained_variance_ratio, out_png):
    fig, axes = plt.subplots(1, 2, figsize=(12, 5), sharex=True, sharey=True)

    for ax, model_name in zip(axes, ["finetune", "scratch"]):
        sub_post = df_plot[(df_plot["model"] == model_name) & (df_plot["source"] == "posterior")]
        sub_prior = df_plot[(df_plot["model"] == model_name) & (df_plot["source"] == "prior")]

        ax.scatter(
            sub_prior["pc1"], sub_prior["pc2"],
            s=16, alpha=0.35, label="Prior samples"
        )
        ax.scatter(
            sub_post["pc1"], sub_post["pc2"],
            s=20, alpha=0.8, label="Posterior means"
        )

        ax.set_title(model_name)
        ax.set_xlabel(f"PC1 ({explained_variance_ratio[0]*100:.1f}%)")
        ax.set_ylabel(f"PC2 ({explained_variance_ratio[1]*100:.1f}%)")
        ax.legend()

    plt.suptitle("Prior samples vs Posterior means for the same antigen")
    plt.tight_layout()
    plt.savefig(out_png, dpi=200, bbox_inches="tight")
    plt.close()


def main():
    if not os.path.exists(FINETUNE_CKPT):
        raise FileNotFoundError(f"Not found: {FINETUNE_CKPT}")
    if not os.path.exists(SCRATCH_CKPT):
        raise FileNotFoundError(f"Not found: {SCRATCH_CKPT}")

    df_sub = get_target_rows(CSV_PATH, ANTIGEN_COL, CDR3_COL, TARGET_ANTIGEN)
    if len(df_sub) == 0:
        raise ValueError("No exact-match rows found for TARGET_ANTIGEN in the CSV.")

    print(f"Found {len(df_sub)} real samples for the target antigen.")

    model_finetune, config_finetune, max_antigen_len_finetune = load_model(FINETUNE_CKPT)
    model_scratch, config_scratch, max_antigen_len_scratch = load_model(SCRATCH_CKPT)

    posterior_finetune = extract_posterior_means(
        model_finetune, config_finetune, max_antigen_len_finetune, df_sub
    )
    prior_finetune = sample_prior(
        model_finetune, max_antigen_len_finetune, TARGET_ANTIGEN, NUM_PRIOR_SAMPLES
    )

    posterior_scratch = extract_posterior_means(
        model_scratch, config_scratch, max_antigen_len_scratch, df_sub
    )
    prior_scratch = sample_prior(
        model_scratch, max_antigen_len_scratch, TARGET_ANTIGEN, NUM_PRIOR_SAMPLES
    )

    df_plot, evr = build_projection_dataframe(
        posterior_finetune,
        prior_finetune,
        posterior_scratch,
        prior_scratch,
    )

    df_plot.to_csv(OUT_CSV, index=False)
    plot_projection(df_plot, evr, OUT_PNG)

    print(f"Saved projection data to: {OUT_CSV}")
    print(f"Saved figure to: {OUT_PNG}")


if __name__ == "__main__":
    main()

