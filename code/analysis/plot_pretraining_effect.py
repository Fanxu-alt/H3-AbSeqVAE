import os
import math
import random
from collections import Counter
from dataclasses import dataclass
from typing import Dict, List, Tuple

import matplotlib
matplotlib.use("Agg")

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from matplotlib import gridspec
from matplotlib.colors import LinearSegmentedColormap, TwoSlopeNorm
from matplotlib.patches import Rectangle

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader, random_split

@dataclass
class PlotConfig:
    csv_path: str = "CoV-AbDab.csv"
    antigen_col: str = "antigen"
    cdr3_col: str = "cdr3"

    scratch_ckpt: str = "conditional_cvae_scratch.pt"
    finetune_ckpt: str = "conditional_cvae_finetune.pt"

    outdir: str = "pretraining_effect_plots"
    batch_size: int = 256
    val_fraction: float = 0.10
    split_seed: int = 7
    eval_seed: int = 7

    num_generation_samples_per_antigen: int = 4
    generation_temperature: float = 0.90
    generation_sample_mode: str = "sample"   # "sample" or "argmax"
    generation_min_len: int = 8

    num_example_rows: int = 6
    dpi: int = 300


cfg = PlotConfig()

AMINO_ACIDS = list("ACDEFGHIKLMNPQRSTVWY")
SPECIAL_TOKENS = ["X", "<PAD>"]
itos = AMINO_ACIDS + SPECIAL_TOKENS
stoi = {ch: i for i, ch in enumerate(itos)}

PAD_IDX = stoi["<PAD>"]
UNK_IDX = stoi["X"]
VOCAB_SIZE = len(itos)

MOTIFS = ["AR", "GG", "YY", "FDY", "GMD", "RG", "YW", "WG"]
E_BLUE = "#4C78A8"
E_RED = "#D95F02"
E_TEAL = "#1B9E77"
E_GOLD = "#E6AB02"
E_PURPLE = "#7570B3"
E_GREY = "#666666"
E_LIGHT = "#F6F6F6"
E_DARK = "#222222"

DIVERGING_CMAP = LinearSegmentedColormap.from_list(
    "e_div", ["#3B4CC0", "#F7F7F7", "#B40426"]
)
SEQUENTIAL_CMAP = LinearSegmentedColormap.from_list(
    "e_seq", ["#F7FBFF", "#6BAED6", "#08306B"]
)

plt.rcParams.update(
    {
        "font.family": "DejaVu Sans",
        "font.size": 8,
        "axes.titlesize": 9,
        "axes.labelsize": 8,
        "xtick.labelsize": 7,
        "ytick.labelsize": 7,
        "legend.fontsize": 7,
        "figure.titlesize": 10,
        "axes.linewidth": 0.6,
        "xtick.major.width": 0.6,
        "ytick.major.width": 0.6,
        "xtick.major.size": 2.5,
        "ytick.major.size": 2.5,
        "savefig.bbox": "tight",
        "savefig.transparent": False,
    }
)

def set_seed(seed: int):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)


def clean_axis(ax):
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)
    for side in ["left", "bottom"]:
        ax.spines[side].set_color("#444444")
        ax.spines[side].set_linewidth(0.6)


def panel_label(ax, label: str):
    ax.text(
        -0.12,
        1.07,
        label,
        transform=ax.transAxes,
        fontsize=11,
        fontweight="bold",
        va="bottom",
        ha="left",
        color=NATURE_DARK,
    )


def save_fig(fig, png_path: str, pdf_path: str, dpi: int):
    fig.savefig(png_path, dpi=dpi, facecolor="white", bbox_inches="tight")
    fig.savefig(pdf_path, dpi=dpi, facecolor="white", bbox_inches="tight")
    plt.close(fig)


def encode_string(seq: str, fixed_len: int) -> Tuple[torch.Tensor, torch.Tensor]:
    true_len = min(len(seq), fixed_len)
    seq = seq[:fixed_len]
    ids = [stoi.get(ch, UNK_IDX) for ch in seq]
    if len(ids) < fixed_len:
        ids += [PAD_IDX] * (fixed_len - len(ids))
    return torch.tensor(ids, dtype=torch.long), torch.tensor(true_len, dtype=torch.long)


def decode_tokens(token_ids: List[int]) -> str:
    chars = []
    for idx in token_ids:
        ch = itos[idx]
        if ch == "<PAD>":
            continue
        chars.append(ch)
    return "".join(chars)


def aa_composition_from_sequences(seqs: List[str]) -> Dict[str, float]:
    total = Counter()
    n = 0
    for s in seqs:
        total.update([ch for ch in s if ch in AMINO_ACIDS])
        n += sum(1 for ch in s if ch in AMINO_ACIDS)
    n = max(n, 1)
    return {aa: total.get(aa, 0) / n for aa in AMINO_ACIDS}


def motif_prevalence(seqs: List[str], motifs: List[str]) -> Dict[str, float]:
    prev = {}
    N = max(len(seqs), 1)
    for motif in motifs:
        cnt = 0
        for s in seqs:
            if motif in s:
                cnt += 1
        prev[motif] = cnt / N
    return prev


def js_divergence_from_counts(p: np.ndarray, q: np.ndarray) -> float:
    p = np.asarray(p, dtype=float)
    q = np.asarray(q, dtype=float)
    p = p / max(p.sum(), 1e-12)
    q = q / max(q.sum(), 1e-12)
    m = 0.5 * (p + q)

    def kl(a, b):
        mask = (a > 0) & (b > 0)
        return np.sum(a[mask] * np.log2(a[mask] / b[mask]))

    return 0.5 * kl(p, m) + 0.5 * kl(q, m)


def safe_mean(xs):
    return float(np.mean(xs)) if len(xs) > 0 else float("nan")


def truncate_middle(s: str, max_len: int = 48) -> str:
    if len(s) <= max_len:
        return s
    keep = max_len - 3
    left = keep // 2
    right = keep - left
    return s[:left] + "..." + s[-right:]

class AntigenCDR3Dataset(Dataset):
    def __init__(self, csv_path, antigen_col, cdr3_col):
        df = pd.read_csv(csv_path)

        if antigen_col not in df.columns:
            raise ValueError(f"Column '{antigen_col}' not found in {csv_path}")
        if cdr3_col not in df.columns:
            raise ValueError(f"Column '{cdr3_col}' not found in {csv_path}")

        df = df[[antigen_col, cdr3_col]].dropna().copy()
        df[antigen_col] = df[antigen_col].astype(str).str.strip().str.upper()
        df[cdr3_col] = df[cdr3_col].astype(str).str.strip().str.upper()
        df = df[(df[antigen_col].str.len() > 0) & (df[cdr3_col].str.len() > 0)].reset_index(drop=True)

        if len(df) == 0:
            raise ValueError("No valid rows after filtering.")

        self.antigen_col = antigen_col
        self.cdr3_col = cdr3_col
        self.df = df
        self.samples = list(zip(df[antigen_col].tolist(), df[cdr3_col].tolist()))
        self.max_antigen_len = max(len(a) for a, _ in self.samples)

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, idx):
        antigen, cdr3 = self.samples[idx]
        return {
            "index": idx,
            "antigen": antigen,
            "cdr3": cdr3,
        }


class EncodedSubset(Dataset):
    def __init__(self, dataset: AntigenCDR3Dataset, indices: List[int], max_cdr3_len: int):
        self.dataset = dataset
        self.indices = list(indices)
        self.max_cdr3_len = max_cdr3_len
        self.max_antigen_len = dataset.max_antigen_len

    def __len__(self):
        return len(self.indices)

    def __getitem__(self, i):
        idx = self.indices[i]
        item = self.dataset[idx]
        antigen = item["antigen"]
        cdr3 = item["cdr3"]

        a, a_len = encode_string(antigen, self.max_antigen_len)
        x, x_len = encode_string(cdr3, self.max_cdr3_len)
        a_mask = (a != PAD_IDX).long()

        return {
            "dataset_index": idx,
            "antigen_str": antigen,
            "cdr3_str": cdr3,
            "x": x,
            "x_len": x_len,
            "a": a,
            "a_mask": a_mask,
            "a_len": a_len,
        }


def collate_fn(batch):
    return {
        "dataset_index": torch.tensor([b["dataset_index"] for b in batch], dtype=torch.long),
        "antigen_str": [b["antigen_str"] for b in batch],
        "cdr3_str": [b["cdr3_str"] for b in batch],
        "x": torch.stack([b["x"] for b in batch], dim=0),
        "x_len": torch.stack([b["x_len"] for b in batch], dim=0),
        "a": torch.stack([b["a"] for b in batch], dim=0),
        "a_mask": torch.stack([b["a_mask"] for b in batch], dim=0),
        "a_len": torch.stack([b["a_len"] for b in batch], dim=0),
    }

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

    def forward(self, x, a, a_mask):
        hx = self.encoder.encode_feature(x)
        ha = self.antigen_encoder(a, a_mask)

        q_input = torch.cat([hx, ha], dim=-1)
        mu_q = self.posterior_mu(q_input)
        logvar_q = self.posterior_logvar(q_input)

        mu_p = self.prior_mu(ha)
        logvar_p = self.prior_logvar(ha)

        z = self.reparameterize(mu_q, logvar_q)

        z_cond = self.decoder_input(torch.cat([z, ha], dim=-1))
        logits = self.decoder(z_cond)
        len_logits = self.length_head(z_cond)

        return logits, mu_q, logvar_q, mu_p, logvar_p, len_logits, z_cond

    @torch.no_grad()
    def generate_from_antigen(
        self,
        a,
        a_mask,
        num_samples=1,
        min_len=5,
        sample_mode="sample",
        temperature=1.0,
    ):
        ha = self.antigen_encoder(a, a_mask)
        mu_p = self.prior_mu(ha)
        logvar_p = self.prior_logvar(ha)

        results = []
        for _ in range(num_samples):
            z = self.reparameterize(mu_p, logvar_p)
            z_cond = self.decoder_input(torch.cat([z, ha], dim=-1))
            logits = self.decoder(z_cond)
            len_logits = self.length_head(z_cond)

            if temperature != 1.0:
                logits = logits / temperature

            pred_len = torch.argmax(len_logits, dim=-1) + 1
            pred_len = torch.clamp(pred_len, min=min_len, max=logits.size(1))

            if sample_mode == "sample":
                probs = torch.softmax(logits, dim=-1)
                preds = torch.multinomial(
                    probs.reshape(-1, probs.size(-1)), num_samples=1
                ).view(logits.size(0), logits.size(1))
            else:
                preds = logits.argmax(dim=-1)

            results.append((preds, pred_len))
        return results

def load_conditional_model(ckpt_path: str, device: str):
    if not os.path.exists(ckpt_path):
        raise FileNotFoundError(f"Checkpoint not found: {ckpt_path}")

    ckpt = torch.load(ckpt_path, map_location=device)
    if "max_antigen_len" not in ckpt:
        raise ValueError(
            f"{ckpt_path} does not look like a conditional CVAE checkpoint "
            f"(missing 'max_antigen_len'). Use conditional_cvae_scratch.pt / conditional_cvae_finetune.pt."
        )

    model = ConditionalCNNVAE(ckpt["config"]).to(device)
    model.load_state_dict(ckpt["model_state_dict"], strict=True)
    model.eval()
    return model, ckpt

def conditional_vae_loss_components(
    logits,
    targets,
    true_lengths,
    mu_q,
    logvar_q,
    mu_p,
    logvar_p,
    len_logits,
):
    flat_ce = F.cross_entropy(
        logits.reshape(-1, logits.size(-1)),
        targets.reshape(-1),
        ignore_index=PAD_IDX,
        reduction="none",
    ).view(targets.size(0), targets.size(1))

    mask = (targets != PAD_IDX).float()
    token_ce = flat_ce.sum(dim=1) / mask.sum(dim=1).clamp_min(1.0)

    var_q = torch.exp(logvar_q)
    var_p = torch.exp(logvar_p)
    kl = 0.5 * torch.sum(
        logvar_p - logvar_q + (var_q + (mu_q - mu_p).pow(2)) / var_p - 1,
        dim=1
    )

    len_targets = true_lengths - 1
    len_loss = F.cross_entropy(len_logits, len_targets, reduction="none")

    return token_ce, kl, len_loss


@torch.no_grad()
def evaluate_reconstruction(model, loader, device: str) -> Tuple[Dict, pd.DataFrame]:
    model.eval()

    rows = []
    agg_token_ce = []
    agg_kl = []
    agg_len_loss = []
    agg_token_acc = []
    agg_exact = []
    agg_len_mae = []

    for batch in loader:
        x = batch["x"].to(device)
        x_len = batch["x_len"].to(device)
        a = batch["a"].to(device)
        a_mask = batch["a_mask"].to(device)

        logits, mu_q, logvar_q, mu_p, logvar_p, len_logits, _ = model(x, a, a_mask)
        token_ce, kl, len_loss = conditional_vae_loss_components(
            logits, x, x_len, mu_q, logvar_q, mu_p, logvar_p, len_logits
        )

        pred_tokens = logits.argmax(dim=-1)
        pred_len = len_logits.argmax(dim=-1) + 1

        for i in range(x.size(0)):
            true_len_i = int(x_len[i].item())
            pred_len_i = int(pred_len[i].item())

            true_tokens = x[i, :true_len_i].detach().cpu().tolist()
            pred_tokens_i = pred_tokens[i, :pred_len_i].detach().cpu().tolist()

            true_seq = decode_tokens(true_tokens)
            pred_seq = decode_tokens(pred_tokens_i)

            token_mask = x[i] != PAD_IDX
            correct = ((pred_tokens[i] == x[i]) & token_mask).sum().item()
            denom = token_mask.sum().item()
            token_acc = correct / max(denom, 1)

            exact = int(true_seq == pred_seq)
            len_mae = abs(pred_len_i - true_len_i)

            row = {
                "dataset_index": int(batch["dataset_index"][i].item()),
                "antigen": batch["antigen_str"][i],
                "true_seq": batch["cdr3_str"][i],
                "pred_seq": pred_seq,
                "true_len": true_len_i,
                "pred_len": pred_len_i,
                "token_ce": float(token_ce[i].item()),
                "kl": float(kl[i].item()),
                "len_loss": float(len_loss[i].item()),
                "token_acc": float(token_acc),
                "exact_match": int(exact),
                "len_mae": int(len_mae),
            }
            rows.append(row)

            agg_token_ce.append(row["token_ce"])
            agg_kl.append(row["kl"])
            agg_len_loss.append(row["len_loss"])
            agg_token_acc.append(row["token_acc"])
            agg_exact.append(row["exact_match"])
            agg_len_mae.append(row["len_mae"])

    summary = {
        "token_ce": safe_mean(agg_token_ce),
        "kl": safe_mean(agg_kl),
        "len_loss": safe_mean(agg_len_loss),
        "token_acc": safe_mean(agg_token_acc),
        "exact_match": safe_mean(agg_exact),
        "len_mae": safe_mean(agg_len_mae),
        "n": len(rows),
    }
    return summary, pd.DataFrame(rows)


@torch.no_grad()
def evaluate_generation(
    model,
    loader,
    device: str,
    num_samples_per_antigen: int,
    temperature: float,
    sample_mode: str,
    min_len: int,
) -> Tuple[Dict, pd.DataFrame]:
    model.eval()

    rows = []

    for batch in loader:
        a = batch["a"].to(device)
        a_mask = batch["a_mask"].to(device)

        B = a.size(0)
        for i in range(B):
            ai = a[i:i+1]
            a_mask_i = a_mask[i:i+1]
            antigen = batch["antigen_str"][i]
            true_seq = batch["cdr3_str"][i]

            results = model.generate_from_antigen(
                ai,
                a_mask_i,
                num_samples=num_samples_per_antigen,
                min_len=min_len,
                sample_mode=sample_mode,
                temperature=temperature,
            )

            for s_idx, (preds, pred_len) in enumerate(results):
                L = int(pred_len[0].item())
                token_ids = preds[0, :L].detach().cpu().tolist()
                gen_seq = decode_tokens(token_ids)

                rows.append(
                    {
                        "dataset_index": int(batch["dataset_index"][i].item()),
                        "sample_idx": s_idx,
                        "antigen": antigen,
                        "true_seq": true_seq,
                        "gen_seq": gen_seq,
                        "gen_len": len(gen_seq),
                    }
                )

    gen_df = pd.DataFrame(rows)

    real_seqs = loader.dataset.dataset.df.iloc[loader.dataset.indices][cfg.cdr3_col].astype(str).str.upper().tolist()
    real_lengths = [len(s) for s in real_seqs]
    gen_seqs = gen_df["gen_seq"].tolist()
    gen_lengths = gen_df["gen_len"].tolist()

    max_len = max(max(real_lengths), max(gen_lengths) if len(gen_lengths) else 1)
    real_hist = np.bincount(real_lengths, minlength=max_len + 1)
    gen_hist = np.bincount(gen_lengths, minlength=max_len + 1)

    real_comp = aa_composition_from_sequences(real_seqs)
    gen_comp = aa_composition_from_sequences(gen_seqs)

    real_motif = motif_prevalence(real_seqs, MOTIFS)
    gen_motif = motif_prevalence(gen_seqs, MOTIFS)

    aa_comp_l1 = float(sum(abs(gen_comp[aa] - real_comp[aa]) for aa in AMINO_ACIDS))
    motif_l1 = float(sum(abs(gen_motif[m] - real_motif[m]) for m in MOTIFS))
    length_js = float(js_divergence_from_counts(real_hist, gen_hist))
    uniqueness = gen_df["gen_seq"].nunique() / max(len(gen_df), 1)

    summary = {
        "length_js": length_js,
        "aa_comp_l1": aa_comp_l1,
        "motif_l1": motif_l1,
        "uniqueness": uniqueness,
        "n_generated": len(gen_df),
    }

    return summary, gen_df

def build_metric_table(
    scratch_recon: Dict,
    finetune_recon: Dict,
    scratch_gen: Dict,
    finetune_gen: Dict,
) -> pd.DataFrame:
    rows = [
        ("Recon CE ↓", scratch_recon["token_ce"], finetune_recon["token_ce"]),
        ("Posterior KL ↓", scratch_recon["kl"], finetune_recon["kl"]),
        ("Length loss ↓", scratch_recon["len_loss"], finetune_recon["len_loss"]),
        ("Token accuracy ↑", scratch_recon["token_acc"], finetune_recon["token_acc"]),
        ("Exact match ↑", scratch_recon["exact_match"], finetune_recon["exact_match"]),
        ("Length MAE ↓", scratch_recon["len_mae"], finetune_recon["len_mae"]),
        ("Length JS ↓", scratch_gen["length_js"], finetune_gen["length_js"]),
        ("AA comp. L1 ↓", scratch_gen["aa_comp_l1"], finetune_gen["aa_comp_l1"]),
        ("Motif L1 ↓", scratch_gen["motif_l1"], finetune_gen["motif_l1"]),
        ("Uniqueness ↑", scratch_gen["uniqueness"], finetune_gen["uniqueness"]),
    ]
    return pd.DataFrame(rows, columns=["metric", "scratch", "finetune"])


def merge_sample_metrics(scratch_df: pd.DataFrame, finetune_df: pd.DataFrame) -> pd.DataFrame:
    cols = [
        "dataset_index",
        "antigen",
        "true_seq",
        "pred_seq",
        "token_ce",
        "token_acc",
        "exact_match",
        "pred_len",
        "true_len",
        "len_mae",
    ]

    s = scratch_df[cols].copy().rename(
        columns={
            "pred_seq": "pred_seq_scratch",
            "token_ce": "token_ce_scratch",
            "token_acc": "token_acc_scratch",
            "exact_match": "exact_match_scratch",
            "pred_len": "pred_len_scratch",
            "len_mae": "len_mae_scratch",
        }
    )
    f = finetune_df[cols].copy().rename(
        columns={
            "pred_seq": "pred_seq_finetune",
            "token_ce": "token_ce_finetune",
            "token_acc": "token_acc_finetune",
            "exact_match": "exact_match_finetune",
            "pred_len": "pred_len_finetune",
            "len_mae": "len_mae_finetune",
        }
    )

    merged = s.merge(
        f[
            [
                "dataset_index",
                "pred_seq_finetune",
                "token_ce_finetune",
                "token_acc_finetune",
                "exact_match_finetune",
                "pred_len_finetune",
                "len_mae_finetune",
            ]
        ],
        on="dataset_index",
        how="inner",
    )
    merged["token_ce_improvement"] = merged["token_ce_scratch"] - merged["token_ce_finetune"]
    merged["token_acc_improvement"] = merged["token_acc_finetune"] - merged["token_acc_scratch"]
    return merged.sort_values("token_ce_improvement", ascending=False).reset_index(drop=True)

def plot_summary_metrics(ax, metric_df: pd.DataFrame):
    y = np.arange(len(metric_df))
    h = 0.34

    ax.barh(
        y + h / 2,
        metric_df["scratch"].values,
        height=h,
        color="#CFCFCF",
        edgecolor="none",
        label="Scratch",
    )
    ax.barh(
        y - h / 2,
        metric_df["finetune"].values,
        height=h,
        color=NATURE_BLUE,
        edgecolor="none",
        label="Pretrained → finetune",
    )

    ax.set_yticks(y)
    ax.set_yticklabels(metric_df["metric"].tolist())
    ax.invert_yaxis()
    ax.set_title("Validation and generation metrics", pad=8)
    ax.legend(frameon=False, loc="lower right")
    clean_axis(ax)


def plot_paired_scatter(ax, merged_df: pd.DataFrame):
    x = merged_df["token_ce_scratch"].values
    y = merged_df["token_ce_finetune"].values

    lo = min(x.min(), y.min())
    hi = max(x.max(), y.max())
    pad = 0.03 * (hi - lo + 1e-8)

    ax.scatter(
        x,
        y,
        s=11,
        alpha=0.60,
        color=NATURE_PURPLE,
        edgecolors="none",
    )
    ax.plot([lo - pad, hi + pad], [lo - pad, hi + pad], color="#999999", lw=0.9, ls="--")
    ax.set_xlim(lo - pad, hi + pad)
    ax.set_ylim(lo - pad, hi + pad)
    ax.set_xlabel("Scratch token CE")
    ax.set_ylabel("Finetune token CE")
    ax.set_title("Per-sample reconstruction error", pad=8)
    ax.text(
        0.04,
        0.96,
        "Points below diagonal indicate benefit from pretraining",
        transform=ax.transAxes,
        ha="left",
        va="top",
        fontsize=7,
        color=NATURE_GREY,
    )
    clean_axis(ax)


def plot_length_distribution(ax, real_lengths, scratch_lengths, finetune_lengths):
    bins = np.arange(0.5, max(max(real_lengths), max(scratch_lengths), max(finetune_lengths)) + 1.5, 1.0)

    ax.hist(
        real_lengths,
        bins=bins,
        density=True,
        histtype="step",
        linewidth=1.5,
        color=NATURE_DARK,
        label="Real",
    )
    ax.hist(
        scratch_lengths,
        bins=bins,
        density=True,
        histtype="step",
        linewidth=1.2,
        color="#BDBDBD",
        label="Scratch",
    )
    ax.hist(
        finetune_lengths,
        bins=bins,
        density=True,
        histtype="step",
        linewidth=1.2,
        color=NATURE_TEAL,
        label="Pretrained → finetune",
    )

    ax.set_xlabel("Generated CDR3 length")
    ax.set_ylabel("Density")
    ax.set_title("Prior generation length distribution", pad=8)
    ax.legend(frameon=False, loc="best")
    clean_axis(ax)


def plot_aa_deviation_heatmap(ax, real_comp, scratch_comp, finetune_comp):
    mat = np.array(
        [
            [scratch_comp[aa] - real_comp[aa] for aa in AMINO_ACIDS],
            [finetune_comp[aa] - real_comp[aa] for aa in AMINO_ACIDS],
            [
                abs(scratch_comp[aa] - real_comp[aa]) - abs(finetune_comp[aa] - real_comp[aa])
                for aa in AMINO_ACIDS
            ],
        ],
        dtype=float,
    )

    vmax = max(np.max(np.abs(mat)), 0.01)
    im = ax.imshow(
        mat,
        aspect="auto",
        cmap=DIVERGING_CMAP,
        norm=TwoSlopeNorm(vmin=-vmax, vcenter=0.0, vmax=vmax),
    )
    ax.set_yticks([0, 1, 2])
    ax.set_yticklabels(["Scratch - real", "Finetune - real", "Abs. error gain"])
    ax.set_xticks(range(len(AMINO_ACIDS)))
    ax.set_xticklabels(AMINO_ACIDS)
    ax.set_title("Deviation in amino-acid usage", pad=8)
    clean_axis(ax)
    return im


def plot_motif_prevalence_panel(ax, real_prev, scratch_prev, finetune_prev):
    x = np.arange(len(MOTIFS))
    w = 0.25

    ax.bar(x - w, [real_prev[m] for m in MOTIFS], width=w, color=NATURE_DARK, edgecolor="none", label="Real")
    ax.bar(x, [scratch_prev[m] for m in MOTIFS], width=w, color="#CFCFCF", edgecolor="none", label="Scratch")
    ax.bar(x + w, [finetune_prev[m] for m in MOTIFS], width=w, color=NATURE_GOLD, edgecolor="none", label="Pretrained → finetune")

    ax.set_xticks(x)
    ax.set_xticklabels(MOTIFS)
    ax.set_ylabel("Fraction of sequences containing motif")
    ax.set_title("Motif prevalence under prior generation", pad=8)
    ax.legend(frameon=False, ncol=3, loc="upper right")
    clean_axis(ax)


def plot_examples_table(ax, merged_df: pd.DataFrame, n_rows: int):
    ax.axis("off")
    ax.set_title("Representative validation examples", pad=8)

    show_df = merged_df.head(n_rows).copy()

    y0 = 0.96
    line_h = 0.12

    headers = ["Antigen", "True", "Scratch recon", "Finetune recon"]
    xs = [0.00, 0.29, 0.49, 0.74]

    for x, h in zip(xs, headers):
        ax.text(x, y0, h, transform=ax.transAxes, fontweight="bold")

    ax.plot([0.0, 1.0], [y0 - 0.03, y0 - 0.03], transform=ax.transAxes, color="#BBBBBB", lw=0.8)

    for i, (_, row) in enumerate(show_df.iterrows()):
        y = y0 - (i + 1) * line_h
        bg = NATURE_LIGHT if i % 2 == 0 else "white"
        ax.add_patch(Rectangle((0.0, y - 0.04), 1.0, 0.085, transform=ax.transAxes, color=bg, ec="none"))

        antigen = truncate_middle(row["antigen"], 34)
        true_seq = truncate_middle(row["true_seq"], 22)
        scratch_seq = truncate_middle(row["pred_seq_scratch"], 22)
        finetune_seq = truncate_middle(row["pred_seq_finetune"], 22)

        ax.text(xs[0], y, antigen, transform=ax.transAxes, va="center", family="DejaVu Sans Mono", fontsize=7)
        ax.text(xs[1], y, true_seq, transform=ax.transAxes, va="center", family="DejaVu Sans Mono", fontsize=7)
        ax.text(xs[2], y, scratch_seq, transform=ax.transAxes, va="center", family="DejaVu Sans Mono", fontsize=7)
        ax.text(xs[3], y, finetune_seq, transform=ax.transAxes, va="center", family="DejaVu Sans Mono", fontsize=7)


def make_main_figure(
    metric_df: pd.DataFrame,
    merged_df: pd.DataFrame,
    real_lengths: List[int],
    scratch_gen_df: pd.DataFrame,
    finetune_gen_df: pd.DataFrame,
    real_comp: Dict[str, float],
    scratch_comp: Dict[str, float],
    finetune_comp: Dict[str, float],
    real_motif: Dict[str, float],
    scratch_motif: Dict[str, float],
    finetune_motif: Dict[str, float],
    out_png: str,
    out_pdf: str,
    dpi: int,
):
    fig = plt.figure(figsize=(12.6, 8.8), constrained_layout=False)
    gs = gridspec.GridSpec(
        nrows=3,
        ncols=3,
        figure=fig,
        width_ratios=[1.15, 1.00, 1.00],
        height_ratios=[1.05, 1.00, 0.95],
        wspace=0.42,
        hspace=0.52,
    )

    axA = fig.add_subplot(gs[0:2, 0])
    axB = fig.add_subplot(gs[0, 1])
    axC = fig.add_subplot(gs[0, 2])
    axD = fig.add_subplot(gs[1, 1:])
    axE = fig.add_subplot(gs[2, 0:2])
    axF = fig.add_subplot(gs[2, 2])

    plot_summary_metrics(axA, metric_df)
    panel_label(axA, "a")

    plot_paired_scatter(axB, merged_df)
    panel_label(axB, "b")

    plot_length_distribution(
        axC,
        real_lengths,
        scratch_gen_df["gen_len"].tolist(),
        finetune_gen_df["gen_len"].tolist(),
    )
    panel_label(axC, "c")

    im = plot_aa_deviation_heatmap(axD, real_comp, scratch_comp, finetune_comp)
    cbar = fig.colorbar(im, ax=axD, fraction=0.02, pad=0.02)
    cbar.set_label("Deviation from real", rotation=90)
    panel_label(axD, "d")

    plot_motif_prevalence_panel(axE, real_motif, scratch_motif, finetune_motif)
    panel_label(axE, "e")

    plot_examples_table(axF, merged_df, cfg.num_example_rows)
    panel_label(axF, "f")

    fig.suptitle(
        "Pretraining improves downstream antigen-conditioned CDR3 modeling",
        y=0.995,
        fontsize=11,
        fontweight="bold",
    )

    fig.text(
        0.01,
        -0.01,
        "Figure style uses restrained colour, thin axes, and dense panels to approximate a Nature-like aesthetic. "
        "Interpretation should be based on the held-out split evaluated here rather than the training split used during fitting.",
        fontsize=7,
        color=NATURE_GREY,
    )

    save_fig(fig, out_png, out_pdf, dpi)


def export_individual_panels(
    metric_df: pd.DataFrame,
    merged_df: pd.DataFrame,
    real_lengths: List[int],
    scratch_gen_df: pd.DataFrame,
    finetune_gen_df: pd.DataFrame,
    real_comp: Dict[str, float],
    scratch_comp: Dict[str, float],
    finetune_comp: Dict[str, float],
    real_motif: Dict[str, float],
    scratch_motif: Dict[str, float],
    finetune_motif: Dict[str, float],
    outdir: str,
    dpi: int,
):
    # a
    fig, ax = plt.subplots(figsize=(5.8, 6.2))
    plot_summary_metrics(ax, metric_df)
    panel_label(ax, "a")
    save_fig(
        fig,
        os.path.join(outdir, "panel_a_summary_metrics.png"),
        os.path.join(outdir, "panel_a_summary_metrics.pdf"),
        dpi,
    )

    # b
    fig, ax = plt.subplots(figsize=(4.6, 4.2))
    plot_paired_scatter(ax, merged_df)
    panel_label(ax, "b")
    save_fig(
        fig,
        os.path.join(outdir, "panel_b_paired_scatter.png"),
        os.path.join(outdir, "panel_b_paired_scatter.pdf"),
        dpi,
    )

    # c
    fig, ax = plt.subplots(figsize=(4.8, 4.0))
    plot_length_distribution(
        ax,
        real_lengths,
        scratch_gen_df["gen_len"].tolist(),
        finetune_gen_df["gen_len"].tolist(),
    )
    panel_label(ax, "c")
    save_fig(
        fig,
        os.path.join(outdir, "panel_c_length_distribution.png"),
        os.path.join(outdir, "panel_c_length_distribution.pdf"),
        dpi,
    )

    # d
    fig, ax = plt.subplots(figsize=(9.0, 2.8))
    im = plot_aa_deviation_heatmap(ax, real_comp, scratch_comp, finetune_comp)
    cbar = fig.colorbar(im, ax=ax, fraction=0.03, pad=0.02)
    cbar.set_label("Deviation from real", rotation=90)
    panel_label(ax, "d")
    save_fig(
        fig,
        os.path.join(outdir, "panel_d_aa_deviation_heatmap.png"),
        os.path.join(outdir, "panel_d_aa_deviation_heatmap.pdf"),
        dpi,
    )

    # e
    fig, ax = plt.subplots(figsize=(7.6, 3.8))
    plot_motif_prevalence_panel(ax, real_motif, scratch_motif, finetune_motif)
    panel_label(ax, "e")
    save_fig(
        fig,
        os.path.join(outdir, "panel_e_motif_prevalence.png"),
        os.path.join(outdir, "panel_e_motif_prevalence.pdf"),
        dpi,
    )

    # f
    fig, ax = plt.subplots(figsize=(9.0, 4.1))
    plot_examples_table(ax, merged_df, cfg.num_example_rows)
    panel_label(ax, "f")
    save_fig(
        fig,
        os.path.join(outdir, "panel_f_examples.png"),
        os.path.join(outdir, "panel_f_examples.pdf"),
        dpi,
    )

def main():
    os.makedirs(cfg.outdir, exist_ok=True)
    set_seed(cfg.split_seed)

    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"Using device: {device}")

    dataset = AntigenCDR3Dataset(cfg.csv_path, cfg.antigen_col, cfg.cdr3_col)

    scratch_model, scratch_ckpt = load_conditional_model(cfg.scratch_ckpt, device)
    finetune_model, finetune_ckpt = load_conditional_model(cfg.finetune_ckpt, device)

    max_cdr3_len_s = scratch_ckpt["config"]["max_cdr3_len"]
    max_cdr3_len_f = finetune_ckpt["config"]["max_cdr3_len"]
    if max_cdr3_len_s != max_cdr3_len_f:
        raise ValueError(
            f"Scratch and finetune checkpoints have different max_cdr3_len: "
            f"{max_cdr3_len_s} vs {max_cdr3_len_f}"
        )
    max_cdr3_len = max_cdr3_len_s

    indices = list(range(len(dataset)))
    val_size = max(1, int(cfg.val_fraction * len(dataset)))
    train_size = len(dataset) - val_size
    generator = torch.Generator().manual_seed(cfg.split_seed)
    train_subset, val_subset = random_split(indices, [train_size, val_size], generator=generator)

    val_encoded = EncodedSubset(dataset, val_subset.indices, max_cdr3_len=max_cdr3_len)
    val_loader = DataLoader(
        val_encoded,
        batch_size=cfg.batch_size,
        shuffle=False,
        drop_last=False,
        collate_fn=collate_fn,
    )

    print(f"Dataset size: {len(dataset)}")
    print(f"Validation size used for comparison: {len(val_encoded)}")

    set_seed(cfg.eval_seed)
    scratch_recon_summary, scratch_recon_df = evaluate_reconstruction(scratch_model, val_loader, device)

    set_seed(cfg.eval_seed)
    finetune_recon_summary, finetune_recon_df = evaluate_reconstruction(finetune_model, val_loader, device)

    set_seed(cfg.eval_seed)
    scratch_gen_summary, scratch_gen_df = evaluate_generation(
        scratch_model,
        val_loader,
        device=device,
        num_samples_per_antigen=cfg.num_generation_samples_per_antigen,
        temperature=cfg.generation_temperature,
        sample_mode=cfg.generation_sample_mode,
        min_len=cfg.generation_min_len,
    )

    set_seed(cfg.eval_seed)
    finetune_gen_summary, finetune_gen_df = evaluate_generation(
        finetune_model,
        val_loader,
        device=device,
        num_samples_per_antigen=cfg.num_generation_samples_per_antigen,
        temperature=cfg.generation_temperature,
        sample_mode=cfg.generation_sample_mode,
        min_len=cfg.generation_min_len,
    )

    metric_df = build_metric_table(
        scratch_recon_summary,
        finetune_recon_summary,
        scratch_gen_summary,
        finetune_gen_summary,
    )
    merged_df = merge_sample_metrics(scratch_recon_df, finetune_recon_df)

    # Real / generated corpus stats
    real_seqs = [dataset[i]["cdr3"] for i in val_subset.indices]
    real_lengths = [len(s) for s in real_seqs]
    real_comp = aa_composition_from_sequences(real_seqs)
    scratch_comp = aa_composition_from_sequences(scratch_gen_df["gen_seq"].tolist())
    finetune_comp = aa_composition_from_sequences(finetune_gen_df["gen_seq"].tolist())

    real_motif = motif_prevalence(real_seqs, MOTIFS)
    scratch_motif = motif_prevalence(scratch_gen_df["gen_seq"].tolist(), MOTIFS)
    finetune_motif = motif_prevalence(finetune_gen_df["gen_seq"].tolist(), MOTIFS)

    # Save tables
    metric_df.to_csv(os.path.join(cfg.outdir, "summary_metrics.csv"), index=False)
    scratch_recon_df.to_csv(os.path.join(cfg.outdir, "scratch_reconstruction_per_sample.csv"), index=False)
    finetune_recon_df.to_csv(os.path.join(cfg.outdir, "finetune_reconstruction_per_sample.csv"), index=False)
    scratch_gen_df.to_csv(os.path.join(cfg.outdir, "scratch_generation_samples.csv"), index=False)
    finetune_gen_df.to_csv(os.path.join(cfg.outdir, "finetune_generation_samples.csv"), index=False)
    merged_df.to_csv(os.path.join(cfg.outdir, "paired_reconstruction_comparison.csv"), index=False)

    # Main figure
    make_main_figure(
        metric_df=metric_df,
        merged_df=merged_df,
        real_lengths=real_lengths,
        scratch_gen_df=scratch_gen_df,
        finetune_gen_df=finetune_gen_df,
        real_comp=real_comp,
        scratch_comp=scratch_comp,
        finetune_comp=finetune_comp,
        real_motif=real_motif,
        scratch_motif=scratch_motif,
        finetune_motif=finetune_motif,
        out_png=os.path.join(cfg.outdir, "pretraining_effect_nature_style.png"),
        out_pdf=os.path.join(cfg.outdir, "pretraining_effect_nature_style.pdf"),
        dpi=cfg.dpi,
    )

    # Individual panels
    export_individual_panels(
        metric_df=metric_df,
        merged_df=merged_df,
        real_lengths=real_lengths,
        scratch_gen_df=scratch_gen_df,
        finetune_gen_df=finetune_gen_df,
        real_comp=real_comp,
        scratch_comp=scratch_comp,
        finetune_comp=finetune_comp,
        real_motif=real_motif,
        scratch_motif=scratch_motif,
        finetune_motif=finetune_motif,
        outdir=cfg.outdir,
        dpi=cfg.dpi,
    )

    print("\n===== Reconstruction summary =====")
    print("Scratch:")
    for k, v in scratch_recon_summary.items():
        print(f"  {k}: {v:.6f}" if isinstance(v, float) else f"  {k}: {v}")
    print("Pretrained → finetune:")
    for k, v in finetune_recon_summary.items():
        print(f"  {k}: {v:.6f}" if isinstance(v, float) else f"  {k}: {v}")

    print("\n===== Prior generation summary =====")
    print("Scratch:")
    for k, v in scratch_gen_summary.items():
        print(f"  {k}: {v:.6f}" if isinstance(v, float) else f"  {k}: {v}")
    print("Pretrained → finetune:")
    for k, v in finetune_gen_summary.items():
        print(f"  {k}: {v:.6f}" if isinstance(v, float) else f"  {k}: {v}")

    print(f"\nAll outputs saved to: {cfg.outdir}")


if __name__ == "__main__":
    main()
