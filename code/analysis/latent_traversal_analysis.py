import os
import csv
import math
from collections import Counter
from typing import Dict, List, Tuple

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from matplotlib import gridspec
from matplotlib.colors import LinearSegmentedColormap, TwoSlopeNorm
from matplotlib.patches import Rectangle

import torch
import torch.nn as nn
import torch.nn.functional as F

CKPT_PATH = "conditional_cvae_finetune.pt"

TARGET_ANTIGEN = (
    "RVQPTESIVRFPNITNLCPFGEVFNATRFASVYAWNRKRISNCVADYSVLYNSASFSTFKCYGVSPTKLNDLCFTNVYADSFVIRGDEVRQIAPGQTGKIADYNYKLPDDFTGCVIAWNSNNLDSKVGGNYNYLYRLFRKSNLKPFERDISTEIYQAGSTPCNGVEGFNCYFPLQSYGFQPTNGVGYQPYRVVVLSFELLHAPATVCGPKKSTNLVKNKCVNF"
)

DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
SEED = 7

# Traversal settings
DELTA_VALUES = [-3.0, -2.0, -1.0, 0.0, 1.0, 2.0, 3.0]
SAMPLE_MODE = "sample"      # "argmax" or "sample"
TEMPERATURE = 0.90
MIN_LEN = 8
NUM_SAMPLES_PER_POINT = 32
TRAVERSAL_SCALE = "std"     # "raw" or "std"
STD_SOURCE = "prior"        # "prior" or "unit"

# Selected metrics to visualize in the main heatmap
FEATURES_FOR_HEATMAP = [
    "pred_len",
    "group_aromatic",
    "group_hydrophobic",
    "group_positive",
    "group_negative",
    "group_glycine",
    "group_proline",
    "group_small",
    "group_charge_balance",
]

# Motifs to quantify
MOTIFS = [
    "AR",
    "GG",
    "YY",
    "FDY",
    "GMD",
    "RG",
    "YW",
    "WG",
]

OUTDIR = "latent_traversal_outputs"
RESULTS_CSV = os.path.join(OUTDIR, "latent_traversal_results.csv")
SUMMARY_CSV = os.path.join(OUTDIR, "latent_traversal_summary.csv")
CONTRAST_CSV = os.path.join(OUTDIR, "latent_traversal_contrast.csv")
TOPDIM_CSV = os.path.join(OUTDIR, "latent_top_dimensions.csv")
FIG_PNG = os.path.join(OUTDIR, "latent_traversal_nature_style.png")
FIG_PDF = os.path.join(OUTDIR, "latent_traversal_nature_style.pdf")

PANEL_A_PNG = os.path.join(OUTDIR, "panel_a_heatmap.png")
PANEL_A_PDF = os.path.join(OUTDIR, "panel_a_heatmap.pdf")

PANEL_B_PNG = os.path.join(OUTDIR, "panel_b_length_profile.png")
PANEL_B_PDF = os.path.join(OUTDIR, "panel_b_length_profile.pdf")

PANEL_C_PNG = os.path.join(OUTDIR, "panel_c_aromatic_profile.png")
PANEL_C_PDF = os.path.join(OUTDIR, "panel_c_aromatic_profile.pdf")

PANEL_D_PNG = os.path.join(OUTDIR, "panel_d_motif_dotmap.png")
PANEL_D_PDF = os.path.join(OUTDIR, "panel_d_motif_dotmap.pdf")

PANEL_E_PNG = os.path.join(OUTDIR, "panel_e_sequence_examples.png")
PANEL_E_PDF = os.path.join(OUTDIR, "panel_e_sequence_examples.pdf")

PANEL_F_PNG = os.path.join(OUTDIR, "panel_f_rank_bar.png")
PANEL_F_PDF = os.path.join(OUTDIR, "panel_f_rank_bar.pdf")
# Figure controls
TOP_K_DIMS = 12
TOP_K_LINE_DIMS = 4
DPI = 300

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

def set_seed(seed: int = 7):
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)


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


def aa_composition(seq: str) -> Dict[str, float]:
    L = max(len(seq), 1)
    counter = Counter(seq)
    comp = {f"aa_{aa}": counter.get(aa, 0) / L for aa in AMINO_ACIDS}
    groups = {
        "hydrophobic": set("AVILMFWY"),
        "polar": set("STNQCY"),
        "positive": set("KRH"),
        "negative": set("DE"),
        "glycine": set("G"),
        "proline": set("P"),
        "aromatic": set("FWY"),
        "small": set("AGSTC"),
    }
    for name, aa_set in groups.items():
        count = sum(counter.get(aa, 0) for aa in aa_set)
        comp[f"group_{name}"] = count / L
    comp["group_charge_balance"] = comp["group_positive"] - comp["group_negative"]
    comp["n_unique_aa"] = len(counter)
    return comp


def shannon_entropy(seq: str) -> float:
    if not seq:
        return 0.0
    L = len(seq)
    counter = Counter(seq)
    probs = np.array([v / L for v in counter.values()], dtype=float)
    return float(-(probs * np.log2(probs + 1e-12)).sum())


def motif_stats(seq: str, motifs: List[str]) -> Dict[str, int]:
    stats = {}
    for motif in motifs:
        count = 0
        start = 0
        while True:
            idx = seq.find(motif, start)
            if idx == -1:
                break
            count += 1
            start = idx + 1
        stats[f"motif_{motif}_count"] = count
        stats[f"motif_{motif}_present"] = int(count > 0)
    return stats


def save_csv(path: str, rows: List[Dict]):
    if not rows:
        print(f"[WARN] No rows to save: {path}")
        return
    fieldnames = list(rows[0].keys())
    with open(path, "w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)


@torch.no_grad()
def get_antigen_condition(
    model: ConditionalCNNVAE,
    antigen_seq: str,
    antigen_max_len: int,
    device: str,
):
    a = encode_seq(antigen_seq, antigen_max_len).unsqueeze(0).to(device)
    a_mask = (a != PAD_IDX).long()
    ha = model.antigen_encoder(a, a_mask)
    mu_p = model.prior_mu(ha)
    logvar_p = model.prior_logvar(ha)
    return a, a_mask, ha, mu_p, logvar_p


@torch.no_grad()
def decode_from_given_z(
    model: ConditionalCNNVAE,
    z: torch.Tensor,
    ha: torch.Tensor,
    max_cdr3_len: int,
    min_len: int = 8,
    temperature: float = 1.0,
    sample_mode: str = "argmax",
) -> Tuple[str, int, torch.Tensor, torch.Tensor]:
    z_cond = model.decoder_input(torch.cat([z, ha], dim=-1))
    token_logits = model.decoder(z_cond)
    len_logits = model.length_head(z_cond)

    if temperature != 1.0:
        token_logits = token_logits / temperature

    pred_len = torch.argmax(len_logits, dim=-1) + 1
    pred_len = int(torch.clamp(pred_len, min=min_len, max=max_cdr3_len).item())

    if sample_mode == "sample":
        probs = torch.softmax(token_logits, dim=-1)
        preds = torch.multinomial(probs.reshape(-1, probs.size(-1)), num_samples=1).view(1, max_cdr3_len)
    else:
        preds = token_logits.argmax(dim=-1)

    token_ids = preds[0, :pred_len].cpu().tolist()
    seq = decode_tokens(token_ids)
    return seq, pred_len, token_logits, len_logits


def summarize_records(records: List[Dict], motifs: List[str]) -> List[Dict]:
    df = pd.DataFrame(records)
    group_cols = ["latent_dim", "delta"]

    numeric_cols = [
        c for c in df.columns
        if c not in ["sequence"] + group_cols and pd.api.types.is_numeric_dtype(df[c])
    ]

    agg = (
        df.groupby(group_cols, as_index=False)[numeric_cols]
        .mean()
    )

    n_df = (
        df.groupby(group_cols, as_index=False)
        .size()
        .rename(columns={"size": "n"})
    )
    agg = agg.merge(n_df, on=group_cols, how="left")

    seq_div = (
        df.groupby(group_cols, as_index=False)["sequence"]
        .agg(lambda x: len(set(x)) / max(len(x), 1))
        .rename(columns={"sequence": "sequence_uniqueness"})
    )
    agg = agg.merge(seq_div, on=group_cols, how="left")

    rename_map = {}
    for motif in motifs:
        present_key = f"motif_{motif}_present"
        count_key = f"motif_{motif}_count"
        if present_key in agg.columns:
            rename_map[present_key] = f"frac_{present_key}"
        if count_key in agg.columns:
            rename_map[count_key] = f"mean_{count_key}"

    if rename_map:
        agg = agg.rename(columns=rename_map)

    agg = agg.sort_values(["latent_dim", "delta"]).reset_index(drop=True)
    return agg.to_dict(orient="records")

def build_contrast_table(summary_df: pd.DataFrame, motifs: List[str]) -> pd.DataFrame:
    rows = []
    dims = sorted([d for d in summary_df["latent_dim"].unique() if d >= 0])
    for dim_idx in dims:
        sub = summary_df[summary_df["latent_dim"] == dim_idx].copy()
        if -3.0 not in set(sub["delta"]) or 3.0 not in set(sub["delta"]):
            continue
        r_neg = sub[sub["delta"] == -3.0].iloc[0]
        r_pos = sub[sub["delta"] == 3.0].iloc[0]
        out = {"latent_dim": int(dim_idx)}
        for feat in FEATURES_FOR_HEATMAP:
            if feat in sub.columns:
                out[f"delta_{feat}"] = float(r_pos[feat] - r_neg[feat])
        out["effect_l1"] = float(
            sum(abs(out.get(f"delta_{feat}", 0.0)) for feat in FEATURES_FOR_HEATMAP)
        )
        for motif in motifs:
            key = f"frac_motif_{motif}_present"
            if key in sub.columns:
                out[f"delta_motif_{motif}"] = float(r_pos[key] - r_neg[key])
        rows.append(out)
    return pd.DataFrame(rows).sort_values("effect_l1", ascending=False).reset_index(drop=True)


def z_score_by_column(mat: np.ndarray) -> np.ndarray:
    out = mat.copy().astype(float)
    for j in range(out.shape[1]):
        col = out[:, j]
        s = np.nanstd(col)
        m = np.nanmean(col)
        out[:, j] = 0.0 if s < 1e-8 else (col - m) / s
    return out


def clean_axis(ax):
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)
    for side in ["left", "bottom"]:
        ax.spines[side].set_color("#444444")
        ax.spines[side].set_linewidth(0.6)

def save_panel_figure(fig, png_path: str, pdf_path: str):
    fig.savefig(png_path, dpi=DPI, facecolor="white", bbox_inches="tight")
    fig.savefig(pdf_path, dpi=DPI, facecolor="white", bbox_inches="tight")
    plt.close(fig)
def panel_label(ax, label: str):
    ax.text(
        -0.12,
        1.08,
        label,
        transform=ax.transAxes,
        fontsize=11,
        fontweight="bold",
        va="bottom",
        ha="left",
        color=NATURE_DARK,
    )


def feature_display_name(name: str) -> str:
    mapping = {
        "pred_len": "Length",
        "group_aromatic": "Aromatic",
        "group_hydrophobic": "Hydrophobic",
        "group_positive": "Positive",
        "group_negative": "Negative",
        "group_glycine": "Glycine",
        "group_proline": "Proline",
        "group_small": "Small",
        "group_charge_balance": "Charge balance",
    }
    return mapping.get(name, name)


def motif_display_name(name: str) -> str:
    return name


def plot_heatmap(ax, contrast_df: pd.DataFrame, dims: List[int], features: List[str]):
    mat = []
    for dim in dims:
        row = contrast_df[contrast_df["latent_dim"] == dim].iloc[0]
        mat.append([row.get(f"delta_{feat}", np.nan) for feat in features])
    mat = np.array(mat, dtype=float)
    mat_z = z_score_by_column(mat)
    vmax = np.nanmax(np.abs(mat_z))
    vmax = max(vmax, 1.0)
    im = ax.imshow(mat_z, aspect="auto", cmap=DIVERGING_CMAP, norm=TwoSlopeNorm(vmin=-vmax, vcenter=0, vmax=vmax))
    ax.set_xticks(range(len(features)))
    ax.set_xticklabels([feature_display_name(f) for f in features], rotation=45, ha="right")
    ax.set_yticks(range(len(dims)))
    ax.set_yticklabels([f"z{d}" for d in dims])
    ax.set_title("Latent-dimension effect map\n(+3 vs -3; column-wise z-score)", pad=8)
    clean_axis(ax)
    return im


def plot_line_profiles(ax, summary_df: pd.DataFrame, top_dims: List[int], feature: str):
    colors = [NATURE_BLUE, NATURE_RED, NATURE_TEAL, NATURE_PURPLE]
    for i, dim in enumerate(top_dims):
        sub = summary_df[summary_df["latent_dim"] == dim].sort_values("delta")
        ax.plot(
            sub["delta"],
            sub[feature],
            marker="o",
            markersize=2.8,
            linewidth=1.2,
            color=colors[i % len(colors)],
            label=f"z{dim}",
        )
    ax.axvline(0, color="#BBBBBB", lw=0.8, ls="--")
    ax.set_xlabel("Traversal offset")
    ax.set_ylabel(feature_display_name(feature))
    ax.set_title(f"Profiles across traversal offsets\nTop dimensions by global effect", pad=8)
    clean_axis(ax)
    ax.legend(frameon=False, ncol=2, loc="best")


def plot_motif_dotmap(ax, contrast_df: pd.DataFrame, dims: List[int], motifs: List[str]):
    vals = []
    for dim in dims:
        row = contrast_df[contrast_df["latent_dim"] == dim].iloc[0]
        vals.append([row.get(f"delta_motif_{m}", 0.0) for m in motifs])
    vals = np.array(vals, dtype=float)
    vmax = max(np.max(np.abs(vals)), 0.05)
    norm = TwoSlopeNorm(vmin=-vmax, vcenter=0.0, vmax=vmax)
    for i in range(vals.shape[0]):
        for j in range(vals.shape[1]):
            v = vals[i, j]
            size = 20 + 220 * min(abs(v) / vmax, 1.0)
            ax.scatter(j, i, s=size, c=[v], cmap=DIVERGING_CMAP, norm=norm, edgecolors="none")
    ax.set_xlim(-0.7, len(motifs) - 0.3)
    ax.set_ylim(-0.7, len(dims) - 0.3)
    ax.invert_yaxis()
    ax.set_xticks(range(len(motifs)))
    ax.set_xticklabels([motif_display_name(m) for m in motifs], rotation=45, ha="right")
    ax.set_yticks(range(len(dims)))
    ax.set_yticklabels([f"z{d}" for d in dims])
    ax.set_title("Motif prevalence shifts\n(circle size = |effect|, color = direction)", pad=8)
    clean_axis(ax)
    ax.grid(False)
    sm = plt.cm.ScalarMappable(cmap=DIVERGING_CMAP, norm=norm)
    sm.set_array([])
    return sm


def plot_sequence_examples(ax, records_df: pd.DataFrame, top_dims: List[int]):
    ax.axis("off")
    ax.set_title("Representative decoded sequences", pad=8)

    rows = []
    baseline_seq = records_df[records_df["latent_dim"] == -1]["sequence"].mode().iloc[0]
    rows.append(("Baseline", "0", baseline_seq))
    for dim in top_dims[:min(4, len(top_dims))]:
        sub = records_df[(records_df["latent_dim"] == dim) & (records_df["sample_idx"] == 0)]
        neg = sub[sub["delta"] == -3.0]["sequence"]
        pos = sub[sub["delta"] == 3.0]["sequence"]
        rows.append((f"z{dim}", "-3", neg.iloc[0] if len(neg) else "NA"))
        rows.append((f"z{dim}", "+3", pos.iloc[0] if len(pos) else "NA"))

    y0 = 0.96
    line_h = 0.11
    ax.text(0.00, y0, "Dimension", fontweight="bold", transform=ax.transAxes)
    ax.text(0.22, y0, "Offset", fontweight="bold", transform=ax.transAxes)
    ax.text(0.34, y0, "Sequence", fontweight="bold", transform=ax.transAxes)
    ax.plot([0.0, 1.0], [y0 - 0.03, y0 - 0.03], transform=ax.transAxes, color="#BBBBBB", lw=0.8)

    for i, (dim_lab, delta_lab, seq) in enumerate(rows):
        y = y0 - (i + 1) * line_h
        bg = NATURE_LIGHT if i % 2 == 0 else "white"
        ax.add_patch(Rectangle((0.0, y - 0.035), 1.0, 0.08, transform=ax.transAxes, color=bg, ec="none"))
        ax.text(0.00, y, dim_lab, transform=ax.transAxes, va="center")
        ax.text(0.22, y, delta_lab, transform=ax.transAxes, va="center")
        shown = seq if len(seq) <= 54 else (seq[:51] + "...")
        ax.text(0.34, y, shown, transform=ax.transAxes, va="center", family="DejaVu Sans Mono", fontsize=7)


def plot_rank_bar(ax, contrast_df: pd.DataFrame, top_dims: List[int]):
    sub = contrast_df[contrast_df["latent_dim"].isin(top_dims)].copy()
    sub = sub.sort_values("effect_l1", ascending=True)
    ax.barh(
        [f"z{d}" for d in sub["latent_dim"]],
        sub["effect_l1"],
        color=NATURE_BLUE,
        edgecolor="none",
        height=0.65,
    )
    ax.set_xlabel("Aggregate effect score")
    ax.set_title("Dimensions ranked by overall effect size", pad=8)
    clean_axis(ax)
def export_individual_panels(
    summary_df: pd.DataFrame,
    records_df: pd.DataFrame,
    contrast_df: pd.DataFrame,
):
    top_dims = contrast_df["latent_dim"].head(TOP_K_DIMS).tolist()
    line_dims = contrast_df["latent_dim"].head(TOP_K_LINE_DIMS).tolist()

    fig, ax = plt.subplots(figsize=(6.0, 6.8))
    im = plot_heatmap(ax, contrast_df, top_dims, FEATURES_FOR_HEATMAP)
    cbar = fig.colorbar(im, ax=ax, fraction=0.046, pad=0.03)
    cbar.set_label("Standardized effect", rotation=90)
    panel_label(ax, "a")
    save_panel_figure(fig, PANEL_A_PNG, PANEL_A_PDF)

    fig, ax = plt.subplots(figsize=(5.0, 3.6))
    plot_line_profiles(ax, summary_df, line_dims, "pred_len")
    panel_label(ax, "b")
    save_panel_figure(fig, PANEL_B_PNG, PANEL_B_PDF)

    fig, ax = plt.subplots(figsize=(5.0, 3.6))
    plot_line_profiles(ax, summary_df, line_dims, "group_aromatic")
    panel_label(ax, "c")
    save_panel_figure(fig, PANEL_C_PNG, PANEL_C_PDF)

    fig, ax = plt.subplots(figsize=(5.8, 5.6))
    sm = plot_motif_dotmap(ax, contrast_df, top_dims[:8], MOTIFS)
    cbar = fig.colorbar(sm, ax=ax, fraction=0.046, pad=0.03)
    cbar.set_label("Δ motif prevalence", rotation=90)
    panel_label(ax, "d")
    save_panel_figure(fig, PANEL_D_PNG, PANEL_D_PDF)

    fig, ax = plt.subplots(figsize=(10.5, 3.6))
    plot_sequence_examples(ax, records_df, line_dims)
    panel_label(ax, "e")
    save_panel_figure(fig, PANEL_E_PNG, PANEL_E_PDF)

    fig, ax = plt.subplots(figsize=(4.8, 3.8))
    plot_rank_bar(ax, contrast_df, top_dims[:8])
    panel_label(ax, "f")
    save_panel_figure(fig, PANEL_F_PNG, PANEL_F_PDF)

def make_nature_figure(
    summary_df: pd.DataFrame,
    records_df: pd.DataFrame,
    contrast_df: pd.DataFrame,
    output_png: str,
    output_pdf: str,
):
    top_dims = contrast_df["latent_dim"].head(TOP_K_DIMS).tolist()
    line_dims = contrast_df["latent_dim"].head(TOP_K_LINE_DIMS).tolist()

    fig = plt.figure(figsize=(12.2, 8.6), constrained_layout=False)
    gs = gridspec.GridSpec(
        nrows=3,
        ncols=3,
        figure=fig,
        width_ratios=[1.25, 1.0, 0.95],
        height_ratios=[1.15, 0.95, 0.95],
        wspace=0.42,
        hspace=0.50,
    )

    axA = fig.add_subplot(gs[0:2, 0])
    axB = fig.add_subplot(gs[0, 1])
    axC = fig.add_subplot(gs[1, 1])
    axD = fig.add_subplot(gs[0:2, 2])
    axE = fig.add_subplot(gs[2, 0:2])
    axF = fig.add_subplot(gs[2, 2])

    im = plot_heatmap(axA, contrast_df, top_dims, FEATURES_FOR_HEATMAP)
    cbarA = fig.colorbar(im, ax=axA, fraction=0.046, pad=0.02)
    cbarA.set_label("Standardized effect", rotation=90)
    panel_label(axA, "a")

    plot_line_profiles(axB, summary_df, line_dims, "pred_len")
    panel_label(axB, "b")

    plot_line_profiles(axC, summary_df, line_dims, "group_aromatic")
    panel_label(axC, "c")

    sm = plot_motif_dotmap(axD, contrast_df, top_dims[:8], MOTIFS)
    cbarD = fig.colorbar(sm, ax=axD, fraction=0.046, pad=0.02)
    cbarD.set_label("Δ motif prevalence", rotation=90)
    panel_label(axD, "d")

    plot_sequence_examples(axE, records_df, line_dims)
    panel_label(axE, "e")

    plot_rank_bar(axF, contrast_df, top_dims[:8])
    panel_label(axF, "f")

    fig.suptitle(
        "Latent traversal reveals interpretable axes in the conditional CDR3 generator",
        x=0.5,
        y=0.995,
        fontsize=11,
        fontweight="bold",
    )

    fig.text(
        0.01,
        -0.01,
        "Muted palette, thin axes, and dense multi-panel layout are chosen to approximate a Nature-like figure aesthetic; "
        "adapt panel sizes and font family to match your target journal template.",
        fontsize=7,
        color=NATURE_GREY,
    )

    fig.savefig(output_png, dpi=DPI, facecolor="white")
    fig.savefig(output_pdf, dpi=DPI, facecolor="white")
    plt.close(fig)

@torch.no_grad()
def run_latent_traversal():
    set_seed(SEED)
    os.makedirs(OUTDIR, exist_ok=True)

    if not os.path.exists(CKPT_PATH):
        raise FileNotFoundError(f"Checkpoint not found: {CKPT_PATH}")

    ckpt = torch.load(CKPT_PATH, map_location=DEVICE)
    config = ckpt["config"]
    antigen_max_len = ckpt["max_antigen_len"]
    latent_dim = config["latent_dim"]
    max_cdr3_len = config["max_cdr3_len"]

    model = ConditionalCNNVAE(config).to(DEVICE)
    model.load_state_dict(ckpt["model_state_dict"], strict=True)
    model.eval()

    _, _, ha, mu_p, logvar_p = get_antigen_condition(
        model=model,
        antigen_seq=TARGET_ANTIGEN,
        antigen_max_len=antigen_max_len,
        device=DEVICE,
    )

    prior_std = torch.exp(0.5 * logvar_p)
    z_base = mu_p.clone()

    print("=" * 80)
    print("Latent traversal analysis with figure generation")
    print("=" * 80)
    print(f"Device                  : {DEVICE}")
    print(f"Latent dim              : {latent_dim}")
    print(f"Traversal scale         : {TRAVERSAL_SCALE}")
    print(f"Sampling mode           : {SAMPLE_MODE}")
    print(f"Samples per point       : {NUM_SAMPLES_PER_POINT}")
    print(f"Outputs                 : {OUTDIR}")

    records = []

    for dim_idx in [-1] + list(range(latent_dim)):
        deltas = [0.0] if dim_idx == -1 else DELTA_VALUES
        for delta in deltas:
            for sample_idx in range(NUM_SAMPLES_PER_POINT):
                z_mod = z_base.clone()
                if dim_idx >= 0:
                    if TRAVERSAL_SCALE == "std" and STD_SOURCE == "prior":
                        step = float(delta) * prior_std[0, dim_idx].item()
                    elif TRAVERSAL_SCALE == "std":
                        step = float(delta)
                    else:
                        step = float(delta)
                    z_mod[0, dim_idx] += step
                else:
                    step = 0.0

                seq, pred_len, _, _ = decode_from_given_z(
                    model=model,
                    z=z_mod,
                    ha=ha,
                    max_cdr3_len=max_cdr3_len,
                    min_len=MIN_LEN,
                    temperature=TEMPERATURE,
                    sample_mode=SAMPLE_MODE,
                )

                row = {
                    "latent_dim": dim_idx,
                    "delta": float(delta),
                    "delta_applied": float(step),
                    "sample_idx": sample_idx,
                    "pred_len": pred_len,
                    "sequence": seq,
                    "entropy": shannon_entropy(seq),
                }
                row.update(aa_composition(seq))
                row.update(motif_stats(seq, MOTIFS))
                records.append(row)

            if dim_idx == -1:
                example = [r for r in records if r["latent_dim"] == -1 and r["sample_idx"] == 0][0]
                print(f"baseline -> len={example['pred_len']:2d}, seq={example['sequence']}")
            else:
                last = records[-1]
                print(
                    f"dim={dim_idx:02d}, delta={delta:>4} -> len={last['pred_len']:2d}, seq={last['sequence']}"
                )

    summary_rows = summarize_records(records, MOTIFS)
    save_csv(RESULTS_CSV, records)
    save_csv(SUMMARY_CSV, summary_rows)

    records_df = pd.DataFrame(records)
    summary_df = pd.DataFrame(summary_rows)
    contrast_df = build_contrast_table(summary_df, MOTIFS)
    contrast_df.to_csv(CONTRAST_CSV, index=False)
    contrast_df.head(TOP_K_DIMS).to_csv(TOPDIM_CSV, index=False)

    make_nature_figure(
        summary_df=summary_df,
        records_df=records_df,
        contrast_df=contrast_df,
        output_png=FIG_PNG,
        output_pdf=FIG_PDF,
    )

    export_individual_panels(
        summary_df=summary_df,
        records_df=records_df,
        contrast_df=contrast_df,
    )
    print("\nSaved detailed results to :", RESULTS_CSV)
    print("Saved summary results to  :", SUMMARY_CSV)
    print("Saved contrast table to   :", CONTRAST_CSV)
    print("Saved top-dim table to    :", TOPDIM_CSV)
    print("Saved figure PNG to       :", FIG_PNG)
    print("Saved figure PDF to       :", FIG_PDF)

    print("\nTop dimensions by aggregate effect score:")
    for _, row in contrast_df.head(10).iterrows():
        print(f"  z{int(row['latent_dim']):02d} | score={row['effect_l1']:.4f}")


if __name__ == "__main__":
    run_latent_traversal()
