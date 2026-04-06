import os
import math
import random
from dataclasses import dataclass
from collections import Counter
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

@dataclass
class Config:
    csv_path: str = "CoV-AbDab.csv"
    ckpt_path: str = "conditional_cvae_finetune.pt"

    antigen_col: str = "antigen"
    cdr3_col: str = "cdr3"

    target_antigen: str = (
        "RVQPTESIVRFPNITNLCPFDEVFNATRFASVYAWNRKRISNCVADYSVLYNFAPFFAFKCYGVSPTKLNDLCFTNVYADSFVIRGNEVSQIAPGQTGNIADYNYKLPDDFTGCVIAWNSNKLDSKVGGNYNYLYRLFRKSNLKPFERDISTEIYQAGNKPCNGVAGFNCYFPLRSYGFRPTYGVGHQPYRVVVLSFELLHAPATVCGPKKSTNLVKNKCVNF"
    )

    outdir: str = "generated_vs_library_proxy_plots"

    seed: int = 7
    device: str = "cuda" if torch.cuda.is_available() else "cpu"

    num_generate: int = 512
    batch_generate: int = 64
    min_len: int = 8
    sample_mode: str = "sample"   # "sample" or "argmax"
    temperature: float = 0.90

    exact_antigen_match: bool = True

    # proxy settings
    patch_window: int = 4

    dpi: int = 300
    max_examples_to_show: int = 8


cfg = Config()

AMINO_ACIDS = list("ACDEFGHIKLMNPQRSTVWY")
SPECIAL_TOKENS = ["X", "<PAD>"]
itos = AMINO_ACIDS + SPECIAL_TOKENS
stoi = {ch: i for i, ch in enumerate(itos)}

PAD_IDX = stoi["<PAD>"]
UNK_IDX = stoi["X"]
VOCAB_SIZE = len(itos)

E_BLUE = "#4C78A8"
E_RED = "#D95F02"
E_TEAL = "#1B9E77"
E_GOLD = "#E6AB02"
E_PURPLE = "#7570B3"
E_GREY = "#666666"
E_LIGHT = "#F6F6F6"
E_DARK = "#222222"

DIVERGING_CMAP = LinearSegmentedColormap.from_list(
    "nature_div", ["#3B4CC0", "#F7F7F7", "#B40426"]
)
SEQUENTIAL_CMAP = LinearSegmentedColormap.from_list(
    "nature_seq", ["#F7FBFF", "#6BAED6", "#08306B"]
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
        -0.12, 1.07, label,
        transform=ax.transAxes,
        fontsize=11,
        fontweight="bold",
        va="bottom",
        ha="left",
        color=NATURE_DARK,
    )


def save_fig(fig, png_path: str, pdf_path: str):
    fig.savefig(png_path, dpi=cfg.dpi, facecolor="white", bbox_inches="tight")
    fig.savefig(pdf_path, dpi=cfg.dpi, facecolor="white", bbox_inches="tight")
    plt.close(fig)


def encode_seq(seq: str, fixed_len: int) -> torch.Tensor:
    seq = seq[:fixed_len]
    ids = [stoi.get(ch, UNK_IDX) for ch in seq]
    if len(ids) < fixed_len:
        ids += [PAD_IDX] * (fixed_len - len(ids))
    return torch.tensor(ids, dtype=torch.long)


def decode_tokens(token_ids: List[int]) -> str:
    chars = []
    for idx in token_ids:
        ch = itos[idx]
        if ch == "<PAD>":
            continue
        chars.append(ch)
    return "".join(chars)


def truncate_middle(s: str, max_len: int = 36) -> str:
    if len(s) <= max_len:
        return s
    keep = max_len - 3
    left = keep // 2
    right = keep - left
    return s[:left] + "..." + s[-right:]


def bootstrap_ci_diff(x: np.ndarray, y: np.ndarray, n_boot: int = 2000, seed: int = 7):
    rng = np.random.default_rng(seed)
    x = np.asarray(x, dtype=float)
    y = np.asarray(y, dtype=float)
    diffs = []
    for _ in range(n_boot):
        xb = rng.choice(x, size=len(x), replace=True)
        yb = rng.choice(y, size=len(y), replace=True)
        diffs.append(np.median(yb) - np.median(xb))
    lo, hi = np.percentile(diffs, [2.5, 97.5])
    return float(np.median(y) - np.median(x)), float(lo), float(hi)


def rank_biserial_from_u(x: np.ndarray, y: np.ndarray) -> float:
    # no scipy; compute Mann–Whitney U manually
    x = np.asarray(x, dtype=float)
    y = np.asarray(y, dtype=float)
    all_vals = np.concatenate([x, y])
    ranks = pd.Series(all_vals).rank(method="average").to_numpy()
    rx = ranks[:len(x)].sum()
    u_x = rx - len(x) * (len(x) + 1) / 2
    rbc = (2 * u_x) / (len(x) * len(y)) - 1
    return float(rbc)

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
        self, vocab_size, embed_dim, hidden_dim, latent_dim, max_seq_len,
        num_layers=5, kernel_size=3, dropout=0.1
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
        return self.fc_mu(h), self.fc_logvar(h)


class CNNDecoder(nn.Module):
    def __init__(
        self, latent_dim, hidden_dim, vocab_size, max_seq_len,
        num_layers=5, kernel_size=3, dropout=0.1
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
        self, vocab_size, embed_dim, hidden_dim,
        num_layers=3, kernel_size=3, dropout=0.1
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

def load_model(ckpt_path: str, device: str):
    if not os.path.exists(ckpt_path):
        raise FileNotFoundError(f"Checkpoint not found: {ckpt_path}")

    ckpt = torch.load(ckpt_path, map_location=device)
    model = ConditionalCNNVAE(ckpt["config"]).to(device)
    model.load_state_dict(ckpt["model_state_dict"], strict=True)
    model.eval()
    return model, ckpt

def load_antigen_matched_library(csv_path: str, antigen_col: str, cdr3_col: str, target_antigen: str, exact_match: bool):
    df = pd.read_csv(csv_path)
    if antigen_col not in df.columns or cdr3_col not in df.columns:
        raise ValueError(f"CSV must contain columns '{antigen_col}' and '{cdr3_col}'")

    df = df[[antigen_col, cdr3_col]].dropna().copy()
    df[antigen_col] = df[antigen_col].astype(str).str.strip().str.upper()
    df[cdr3_col] = df[cdr3_col].astype(str).str.strip().str.upper()
    df = df[(df[antigen_col].str.len() > 0) & (df[cdr3_col].str.len() > 0)].reset_index(drop=True)

    target = target_antigen.strip().upper()

    if exact_match:
        lib = df[df[antigen_col] == target].copy()
    else:
        lib = df[df[antigen_col].str.contains(target, regex=False)].copy()

    if len(lib) == 0:
        raise ValueError(
            "No antigen-matched library antibodies found in CSV for the target antigen. "
            "Check whether exact antigen sequence matching is appropriate."
        )

    lib = lib.drop_duplicates(subset=[cdr3_col]).reset_index(drop=True)
    return lib

HYDROPHOBIC = set("AVILMFWY")
POSITIVE = set("KRH")
NEGATIVE = set("DE")


def sliding_patch_fraction(seq: str, aa_set: set, window: int) -> float:
    if len(seq) == 0:
        return 0.0
    if len(seq) < window:
        return sum(ch in aa_set for ch in seq) / len(seq)

    vals = []
    for i in range(len(seq) - window + 1):
        chunk = seq[i:i + window]
        vals.append(sum(ch in aa_set for ch in chunk) / window)
    return float(max(vals)) if vals else 0.0


def charge_asymmetry_proxy(seq: str) -> float:
    if len(seq) == 0:
        return 0.0
    mid = len(seq) // 2
    left = seq[:mid]
    right = seq[mid:]

    def net_charge(s):
        pos = sum(ch in POSITIVE for ch in s)
        neg = sum(ch in NEGATIVE for ch in s)
        return pos - neg

    l = net_charge(left)
    r = net_charge(right)
    return abs(l - r) / max(len(seq), 1)


def compute_proxy_metrics(seq: str, patch_window: int) -> Dict[str, float]:
    seq = str(seq).strip().upper()
    L = len(seq)

    return {
        "sequence": seq,
        "Total_CDR_Length": float(L),
        # proxy only
        "PSH_proxy": sliding_patch_fraction(seq, HYDROPHOBIC, patch_window),
        "PPC_proxy": sliding_patch_fraction(seq, POSITIVE, patch_window),
        "PNC_proxy": sliding_patch_fraction(seq, NEGATIVE, patch_window),
        "SFvCSP_proxy": charge_asymmetry_proxy(seq),
        "net_charge": float(sum(ch in POSITIVE for ch in seq) - sum(ch in NEGATIVE for ch in seq)),
        "hydrophobic_fraction": float(sum(ch in HYDROPHOBIC for ch in seq) / max(L, 1)),
    }


def metric_direction_map():
    # higher_is_better here means "user-defined favorable direction"
    # revise these if your design objective is different
    return {
        "Total_CDR_Length": "context",
        "PSH_proxy": "lower",
        "PPC_proxy": "higher",
        "PNC_proxy": "lower",
        "SFvCSP_proxy": "lower",
    }

@torch.no_grad()
def generate_cdr3s_for_antigen(model, ckpt, antigen_seq: str, n_generate: int) -> List[str]:
    antigen_max_len = ckpt["max_antigen_len"]
    a = encode_seq(antigen_seq, antigen_max_len).unsqueeze(0).to(cfg.device)
    a_mask = (a != PAD_IDX).long()

    seqs = []
    remaining = n_generate

    while remaining > 0:
        cur = min(cfg.batch_generate, remaining)
        results = model.generate_from_antigen(
            a.repeat(cur, 1),
            a_mask.repeat(cur, 1),
            num_samples=1,
            min_len=cfg.min_len,
            sample_mode=cfg.sample_mode,
            temperature=cfg.temperature,
        )
        preds, pred_len = results[0]
        for i in range(preds.size(0)):
            L = int(pred_len[i].item())
            token_ids = preds[i, :L].detach().cpu().tolist()
            seqs.append(decode_tokens(token_ids))
        remaining -= cur

    return seqs[:n_generate]

def summarize_group(df: pd.DataFrame, group_name: str) -> pd.DataFrame:
    rows = []
    for metric in ["Total_CDR_Length", "PSH_proxy", "PPC_proxy", "PNC_proxy", "SFvCSP_proxy"]:
        vals = df[metric].to_numpy(dtype=float)
        rows.append(
            {
                "group": group_name,
                "metric": metric,
                "n": len(vals),
                "mean": float(np.mean(vals)),
                "median": float(np.median(vals)),
                "std": float(np.std(vals, ddof=1)) if len(vals) > 1 else 0.0,
                "q25": float(np.percentile(vals, 25)),
                "q75": float(np.percentile(vals, 75)),
            }
        )
    return pd.DataFrame(rows)


def build_comparison_table(lib_df: pd.DataFrame, gen_df: pd.DataFrame) -> pd.DataFrame:
    rows = []
    dirs = metric_direction_map()
    for metric in ["Total_CDR_Length", "PSH_proxy", "PPC_proxy", "PNC_proxy", "SFvCSP_proxy"]:
        x = lib_df[metric].to_numpy(dtype=float)
        y = gen_df[metric].to_numpy(dtype=float)
        delta_median, ci_lo, ci_hi = bootstrap_ci_diff(x, y, n_boot=2000, seed=cfg.seed)
        rbc = rank_biserial_from_u(x, y)

        favorable = None
        if dirs[metric] == "lower":
            favorable = float(np.median(y) < np.median(x))
        elif dirs[metric] == "higher":
            favorable = float(np.median(y) > np.median(x))

        rows.append(
            {
                "metric": metric,
                "library_median": float(np.median(x)),
                "generated_median": float(np.median(y)),
                "median_shift_generated_minus_library": delta_median,
                "median_shift_ci_low": ci_lo,
                "median_shift_ci_high": ci_hi,
                "rank_biserial": rbc,
                "favorable_direction": dirs[metric],
                "favorable_shift": favorable,
            }
        )
    return pd.DataFrame(rows)

DISPLAY_NAMES = {
    "Total_CDR_Length": "Total CDR Length",
    "PSH_proxy": "PSH proxy",
    "PPC_proxy": "PPC proxy",
    "PNC_proxy": "PNC proxy",
    "SFvCSP_proxy": "SFvCSP proxy",
}


def plot_metric_distributions(ax, lib_df: pd.DataFrame, gen_df: pd.DataFrame, metric: str):
    lib = lib_df[metric].to_numpy(dtype=float)
    gen = gen_df[metric].to_numpy(dtype=float)

    parts = ax.violinplot(
        [lib, gen],
        positions=[1, 2],
        showmeans=False,
        showmedians=False,
        showextrema=False,
        widths=0.8,
    )

    for i, body in enumerate(parts["bodies"]):
        body.set_facecolor("#D0D0D0" if i == 0 else NATURE_BLUE)
        body.set_edgecolor("none")
        body.set_alpha(0.65)

    for i, vals in enumerate([lib, gen], start=1):
        q1, med, q3 = np.percentile(vals, [25, 50, 75])
        ax.plot([i, i], [q1, q3], color=NATURE_DARK, lw=1.0)
        ax.scatter([i], [med], color=NATURE_DARK, s=15, zorder=3)

    ax.set_xticks([1, 2])
    ax.set_xticklabels(["Library", "Generated"])
    ax.set_title(DISPLAY_NAMES[metric], pad=6)
    clean_axis(ax)


def plot_shift_heatmap(ax, comp_df: pd.DataFrame):
    metrics = ["Total_CDR_Length", "PSH_proxy", "PPC_proxy", "PNC_proxy", "SFvCSP_proxy"]
    vals = []
    for m in metrics:
        row = comp_df[comp_df["metric"] == m].iloc[0]
        vals.append(row["median_shift_generated_minus_library"])
    mat = np.array(vals, dtype=float).reshape(-1, 1)

    vmax = max(np.max(np.abs(mat)), 1e-6)
    im = ax.imshow(
        mat,
        aspect="auto",
        cmap=DIVERGING_CMAP,
        norm=TwoSlopeNorm(vmin=-vmax, vcenter=0.0, vmax=vmax),
    )
    ax.set_xticks([0])
    ax.set_xticklabels(["Gen - lib"])
    ax.set_yticks(range(len(metrics)))
    ax.set_yticklabels([DISPLAY_NAMES[m] for m in metrics])
    ax.set_title("Median shift summary", pad=8)
    clean_axis(ax)
    return im


def plot_patch_scatter(ax, lib_df: pd.DataFrame, gen_df: pd.DataFrame):
    ax.scatter(
        lib_df["PPC_proxy"], lib_df["PNC_proxy"],
        s=12, alpha=0.50, color="#BDBDBD", edgecolors="none", label="Library"
    )
    ax.scatter(
        gen_df["PPC_proxy"], gen_df["PNC_proxy"],
        s=12, alpha=0.50, color=NATURE_TEAL, edgecolors="none", label="Generated"
    )
    ax.set_xlabel("PPC proxy")
    ax.set_ylabel("PNC proxy")
    ax.set_title("Charge-patch proxy landscape", pad=8)
    ax.legend(frameon=False, loc="best")
    clean_axis(ax)


def plot_length_hist(ax, lib_df: pd.DataFrame, gen_df: pd.DataFrame):
    lib = lib_df["Total_CDR_Length"].astype(int).tolist()
    gen = gen_df["Total_CDR_Length"].astype(int).tolist()
    bins = np.arange(min(lib + gen) - 0.5, max(lib + gen) + 1.5, 1.0)

    ax.hist(lib, bins=bins, density=True, histtype="step", linewidth=1.4, color=NATURE_DARK, label="Library")
    ax.hist(gen, bins=bins, density=True, histtype="step", linewidth=1.4, color=NATURE_RED, label="Generated")
    ax.set_xlabel("CDRH3 length")
    ax.set_ylabel("Density")
    ax.set_title("Length distribution", pad=8)
    ax.legend(frameon=False, loc="best")
    clean_axis(ax)


def plot_examples_table(ax, gen_df: pd.DataFrame):
    ax.axis("off")
    ax.set_title("Representative generated CDRH3 sequences", pad=8)

    show = gen_df.drop_duplicates(subset=["sequence"]).head(cfg.max_examples_to_show).copy()

    y0 = 0.96
    line_h = 0.095
    xs = [0.00, 0.33, 0.46, 0.60, 0.73, 0.86]
    headers = ["Sequence", "Len", "PSH", "PPC", "PNC", "SFvCSP"]

    for x, h in zip(xs, headers):
        ax.text(x, y0, h, transform=ax.transAxes, fontweight="bold")

    ax.plot([0.0, 1.0], [y0 - 0.03, y0 - 0.03], transform=ax.transAxes, color="#BBBBBB", lw=0.8)

    for i, (_, row) in enumerate(show.iterrows()):
        y = y0 - (i + 1) * line_h
        bg = NATURE_LIGHT if i % 2 == 0 else "white"
        ax.add_patch(Rectangle((0.0, y - 0.035), 1.0, 0.07, transform=ax.transAxes, color=bg, ec="none"))

        ax.text(xs[0], y, truncate_middle(row["sequence"], 26), transform=ax.transAxes, va="center", family="DejaVu Sans Mono", fontsize=7)
        ax.text(xs[1], y, f"{row['Total_CDR_Length']:.0f}", transform=ax.transAxes, va="center")
        ax.text(xs[2], y, f"{row['PSH_proxy']:.2f}", transform=ax.transAxes, va="center")
        ax.text(xs[3], y, f"{row['PPC_proxy']:.2f}", transform=ax.transAxes, va="center")
        ax.text(xs[4], y, f"{row['PNC_proxy']:.2f}", transform=ax.transAxes, va="center")
        ax.text(xs[5], y, f"{row['SFvCSP_proxy']:.2f}", transform=ax.transAxes, va="center")


def plot_favorable_bar(ax, comp_df: pd.DataFrame):
    metrics = ["PSH_proxy", "PPC_proxy", "PNC_proxy", "SFvCSP_proxy"]
    vals = []
    labels = []
    for m in metrics:
        row = comp_df[comp_df["metric"] == m].iloc[0]
        direction = row["favorable_direction"]
        shift = row["median_shift_generated_minus_library"]
        if direction == "lower":
            score = -shift
        elif direction == "higher":
            score = shift
        else:
            score = 0.0
        vals.append(score)
        labels.append(DISPLAY_NAMES[m])

    colors = [NATURE_BLUE if v >= 0 else "#C44E52" for v in vals]
    ax.barh(labels, vals, color=colors, edgecolor="none", height=0.65)
    ax.axvline(0, color="#999999", lw=0.8, ls="--")
    ax.set_xlabel("Favorable proxy shift")
    ax.set_title("Direction-of-improvement summary", pad=8)
    clean_axis(ax)


def make_main_figure(lib_df, gen_df, comp_df, out_png, out_pdf):
    fig = plt.figure(figsize=(12.6, 8.8), constrained_layout=False)
    gs = gridspec.GridSpec(
        nrows=3,
        ncols=3,
        figure=fig,
        width_ratios=[1.15, 1.00, 0.95],
        height_ratios=[1.00, 1.00, 0.95],
        wspace=0.42,
        hspace=0.52,
    )

    axA = fig.add_subplot(gs[0, 0])
    axB = fig.add_subplot(gs[0, 1])
    axC = fig.add_subplot(gs[0, 2])
    axD = fig.add_subplot(gs[1, 0])
    axE = fig.add_subplot(gs[1, 1])
    axF = fig.add_subplot(gs[1, 2])
    axG = fig.add_subplot(gs[2, 0])
    axH = fig.add_subplot(gs[2, 1])
    axI = fig.add_subplot(gs[2, 2])

    plot_metric_distributions(axA, lib_df, gen_df, "Total_CDR_Length")
    panel_label(axA, "a")

    plot_metric_distributions(axB, lib_df, gen_df, "PSH_proxy")
    panel_label(axB, "b")

    plot_metric_distributions(axC, lib_df, gen_df, "PPC_proxy")
    panel_label(axC, "c")

    plot_metric_distributions(axD, lib_df, gen_df, "PNC_proxy")
    panel_label(axD, "d")

    plot_metric_distributions(axE, lib_df, gen_df, "SFvCSP_proxy")
    panel_label(axE, "e")

    im = plot_shift_heatmap(axF, comp_df)
    cbar = fig.colorbar(im, ax=axF, fraction=0.046, pad=0.02)
    cbar.set_label("Median shift", rotation=90)
    panel_label(axF, "f")

    plot_patch_scatter(axG, lib_df, gen_df)
    panel_label(axG, "g")

    plot_favorable_bar(axH, comp_df)
    panel_label(axH, "h")

    plot_examples_table(axI, gen_df)
    panel_label(axI, "i")

    fig.suptitle(
        "Generated CDRH3 sequences show favourable shifts in sequence-derived proxy descriptors",
        y=0.995,
        fontsize=11,
        fontweight="bold",
    )

    fig.text(
        0.01,
        -0.01,
        "Important: PSH, PPC, PNC and SFvCSP shown here are sequence-derived proxy descriptors, not structure-based ground-truth scores. "
        "Use a validated scorer or structure pipeline before making definitive developability claims.",
        fontsize=7,
        color=NATURE_GREY,
    )

    save_fig(fig, out_png, out_pdf)


def export_individual_panels(lib_df, gen_df, comp_df, outdir):
    panel_specs = [
        ("panel_a_length", "Total_CDR_Length", (4.3, 3.6)),
        ("panel_b_psh_proxy", "PSH_proxy", (4.3, 3.6)),
        ("panel_c_ppc_proxy", "PPC_proxy", (4.3, 3.6)),
        ("panel_d_pnc_proxy", "PNC_proxy", (4.3, 3.6)),
        ("panel_e_sfv_csp_proxy", "SFvCSP_proxy", (4.3, 3.6)),
    ]

    labels = ["a", "b", "c", "d", "e"]
    for (fname, metric, figsize), label in zip(panel_specs, labels):
        fig, ax = plt.subplots(figsize=figsize)
        plot_metric_distributions(ax, lib_df, gen_df, metric)
        panel_label(ax, label)
        save_fig(
            fig,
            os.path.join(outdir, f"{fname}.png"),
            os.path.join(outdir, f"{fname}.pdf"),
        )

    fig, ax = plt.subplots(figsize=(3.8, 3.8))
    im = plot_shift_heatmap(ax, comp_df)
    cbar = fig.colorbar(im, ax=ax, fraction=0.046, pad=0.02)
    cbar.set_label("Median shift", rotation=90)
    panel_label(ax, "f")
    save_fig(
        fig,
        os.path.join(outdir, "panel_f_shift_heatmap.png"),
        os.path.join(outdir, "panel_f_shift_heatmap.pdf"),
    )

    fig, ax = plt.subplots(figsize=(4.4, 3.8))
    plot_patch_scatter(ax, lib_df, gen_df)
    panel_label(ax, "g")
    save_fig(
        fig,
        os.path.join(outdir, "panel_g_patch_scatter.png"),
        os.path.join(outdir, "panel_g_patch_scatter.pdf"),
    )

    fig, ax = plt.subplots(figsize=(4.6, 3.5))
    plot_favorable_bar(ax, comp_df)
    panel_label(ax, "h")
    save_fig(
        fig,
        os.path.join(outdir, "panel_h_favorable_bar.png"),
        os.path.join(outdir, "panel_h_favorable_bar.pdf"),
    )

    fig, ax = plt.subplots(figsize=(7.2, 4.0))
    plot_examples_table(ax, gen_df)
    panel_label(ax, "i")
    save_fig(
        fig,
        os.path.join(outdir, "panel_i_examples.png"),
        os.path.join(outdir, "panel_i_examples.pdf"),
    )

def main():
    os.makedirs(cfg.outdir, exist_ok=True)
    set_seed(cfg.seed)

    print(f"Using device: {cfg.device}")

    model, ckpt = load_model(cfg.ckpt_path, cfg.device)

    lib = load_antigen_matched_library(
        csv_path=cfg.csv_path,
        antigen_col=cfg.antigen_col,
        cdr3_col=cfg.cdr3_col,
        target_antigen=cfg.target_antigen,
        exact_match=cfg.exact_antigen_match,
    )

    print(f"Matched library antibodies: {len(lib)}")

    gen_seqs = generate_cdr3s_for_antigen(
        model=model,
        ckpt=ckpt,
        antigen_seq=cfg.target_antigen,
        n_generate=cfg.num_generate,
    )

    gen_df = pd.DataFrame({"sequence": gen_seqs}).drop_duplicates().reset_index(drop=True)
    print(f"Unique generated antibodies kept: {len(gen_df)}")

    lib_metrics = lib[cfg.cdr3_col].astype(str).str.upper().apply(lambda s: compute_proxy_metrics(s, cfg.patch_window))
    lib_metrics_df = pd.DataFrame(lib_metrics.tolist())
    lib_metrics_df["source"] = "library"

    gen_metrics = gen_df["sequence"].astype(str).str.upper().apply(lambda s: compute_proxy_metrics(s, cfg.patch_window))
    gen_metrics_df = pd.DataFrame(gen_metrics.tolist())
    gen_metrics_df["source"] = "generated"

    summary_lib = summarize_group(lib_metrics_df, "library")
    summary_gen = summarize_group(gen_metrics_df, "generated")
    comp_df = build_comparison_table(lib_metrics_df, gen_metrics_df)

    summary_all = pd.concat([summary_lib, summary_gen], axis=0, ignore_index=True)

    lib_metrics_df.to_csv(os.path.join(cfg.outdir, "library_proxy_metrics.csv"), index=False)
    gen_metrics_df.to_csv(os.path.join(cfg.outdir, "generated_proxy_metrics.csv"), index=False)
    summary_all.to_csv(os.path.join(cfg.outdir, "proxy_metric_summary.csv"), index=False)
    comp_df.to_csv(os.path.join(cfg.outdir, "proxy_metric_comparison.csv"), index=False)

    make_main_figure(
        lib_df=lib_metrics_df,
        gen_df=gen_metrics_df,
        comp_df=comp_df,
        out_png=os.path.join(cfg.outdir, "generated_vs_library_proxy_nature_style.png"),
        out_pdf=os.path.join(cfg.outdir, "generated_vs_library_proxy_nature_style.pdf"),
    )

    export_individual_panels(
        lib_df=lib_metrics_df,
        gen_df=gen_metrics_df,
        comp_df=comp_df,
        outdir=cfg.outdir,
    )

    print("\nComparison summary:")
    print(comp_df.to_string(index=False))

    print(f"\nOutputs saved to: {cfg.outdir}")


if __name__ == "__main__":
    main()
