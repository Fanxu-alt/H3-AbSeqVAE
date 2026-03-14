import os
import math
import textwrap
from typing import List, Dict

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

# User config

SUMMARY_CSV = "latent_traversal_summary.csv"
RESULTS_CSV = "latent_traversal_results.csv"
OUTDIR = "latent_traversal_plots"

# Select how many of the most representative latent dimensions to draw a length line chart.
TOPK_LENGTH_DIMS = 8
# Which indicators should be included in an amino acid property heatmap?
PROPERTY_METRICS = [
    "mean_length",
    "mean_group_aromatic",
    "mean_group_hydrophobic",
    "mean_group_positive",
    "mean_group_negative",
    "mean_group_glycine",
    "mean_group_proline",
]

# What motifs are drawn in a heatmap?
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
# In a representative sequence diagram, select several dimensions for each attribute.
TOP_DIMS_PER_PROPERTY = 3

REP_DELTAS = [-3.0, 0.0, 3.0]

SEQ_WRAP = 28

# matplotlib dpi
DPI = 200

# Utils

def ensure_outdir(path: str):
    os.makedirs(path, exist_ok=True)


def load_data(summary_csv: str, results_csv: str):
    if not os.path.exists(summary_csv):
        raise FileNotFoundError(f"Missing summary file: {summary_csv}")
    if not os.path.exists(results_csv):
        raise FileNotFoundError(f"Missing results file: {results_csv}")

    summary_df = pd.read_csv(summary_csv)
    results_df = pd.read_csv(results_csv)
# Uniform Type
    summary_df["latent_dim"] = summary_df["latent_dim"].astype(int)
    summary_df["delta"] = summary_df["delta"].astype(float)

    results_df["latent_dim"] = results_df["latent_dim"].astype(int)
    results_df["delta"] = results_df["delta"].astype(float)
    results_df["pred_len"] = results_df["pred_len"].astype(int)
    results_df["sequence"] = results_df["sequence"].astype(str)

    return summary_df, results_df


def get_available_motif_metrics(summary_df: pd.DataFrame, motifs: List[str]) -> List[str]:
    cols = []
    for motif in motifs:
        c = f"frac_motif_{motif}_present"
        if c in summary_df.columns:
            cols.append(c)
    return cols


def get_delta_sorted_values(df: pd.DataFrame) -> List[float]:
    return sorted(df["delta"].dropna().unique().tolist())


def pivot_metric(summary_df: pd.DataFrame, metric: str) -> pd.DataFrame:
    pivot = summary_df.pivot(index="latent_dim", columns="delta", values=metric)
    pivot = pivot.sort_index()
    pivot = pivot.reindex(sorted(pivot.columns), axis=1)
    return pivot


def pick_top_dims_by_range(summary_df: pd.DataFrame, metric: str, topk: int = 8) -> List[int]:
    pivot = pivot_metric(summary_df, metric)
    ranges = (pivot.max(axis=1) - pivot.min(axis=1)).sort_values(ascending=False)
    return ranges.head(topk).index.tolist()


def compute_pos_neg_contrast(summary_df: pd.DataFrame, metric: str, pos_delta: float = 3.0, neg_delta: float = -3.0):
    """
Returns the difference between +3 and -3 for each latent_dim.
    """
    pivot = pivot_metric(summary_df, metric)
    if pos_delta not in pivot.columns or neg_delta not in pivot.columns:
        raise ValueError(f"Missing required deltas for contrast in metric={metric}")
    contrast = pivot[pos_delta] - pivot[neg_delta]
    return contrast.sort_values(ascending=False)


def wrap_seq(seq: str, width: int = 28) -> str:
    return "\n".join(textwrap.wrap(seq, width=width)) if seq else ""


def choose_representative_seq(results_df: pd.DataFrame, latent_dim: int, delta: float) -> Dict:

    sub = results_df[(results_df["latent_dim"] == latent_dim) & (results_df["delta"] == delta)].copy()
    if sub.empty:
        return None

    if "sample_idx" in sub.columns:
        sub = sub.sort_values(["sample_idx", "pred_len"], ascending=[True, True])
    else:
        sub = sub.sort_index()

    row = sub.iloc[0].to_dict()
    return row


def choose_baseline_seq(results_df: pd.DataFrame) -> Dict:
    sub = results_df[(results_df["latent_dim"] == -1) & (results_df["delta"] == 0.0)].copy()
    if sub.empty:
        return None
    if "sample_idx" in sub.columns:
        sub = sub.sort_values(["sample_idx", "pred_len"], ascending=[True, True])
    else:
        sub = sub.sort_index()
    return sub.iloc[0].to_dict()


def draw_simple_heatmap(data: pd.DataFrame, title: str, xlabel: str, ylabel: str, outfile: str,
                        annotate: bool = False, fmt: str = ".2f", figsize=None):
    arr = data.values
    nrows, ncols = arr.shape

    if figsize is None:
        figsize = (max(7, ncols * 1.1), max(6, nrows * 0.35))

    plt.figure(figsize=figsize)
    im = plt.imshow(arr, aspect="auto")
    plt.colorbar(im)

    plt.xticks(ticks=np.arange(ncols), labels=[str(x) for x in data.columns])
    plt.yticks(ticks=np.arange(nrows), labels=[str(y) for y in data.index])

    plt.xlabel(xlabel)
    plt.ylabel(ylabel)
    plt.title(title)

    if annotate:
        for i in range(nrows):
            for j in range(ncols):
                val = arr[i, j]
                text = format(val, fmt)
                plt.text(j, i, text, ha="center", va="center", fontsize=7)

    plt.tight_layout()
    plt.savefig(outfile, dpi=DPI, bbox_inches="tight")
    plt.close()

# Plot 1: Length line chart

def plot_length_lines(summary_df: pd.DataFrame, outdir: str):
    metric = "mean_length"
    top_dims = pick_top_dims_by_range(summary_df, metric=metric, topk=TOPK_LENGTH_DIMS)
    deltas = get_delta_sorted_values(summary_df)

    plt.figure(figsize=(9, 6))
    for dim in top_dims:
        sub = summary_df[summary_df["latent_dim"] == dim].sort_values("delta")
        plt.plot(sub["delta"], sub[metric], marker="o", label=f"dim {dim}")

    plt.xlabel("Delta")
    plt.ylabel("Mean predicted length")
    plt.title("Latent traversal effect on CDRH3 length")
    plt.xticks(deltas)
    plt.legend(ncol=2, fontsize=8)
    plt.tight_layout()

    outfile = os.path.join(outdir, "length_lineplot_topdims.png")
    plt.savefig(outfile, dpi=DPI, bbox_inches="tight")
    plt.close()
    print(f"Saved: {outfile}")

# Plot 2: Amino acid property heatmap

def plot_property_heatmaps(summary_df: pd.DataFrame, outdir: str):
    for metric in PROPERTY_METRICS:
        if metric not in summary_df.columns:
            print(f"[WARN] Skip missing metric: {metric}")
            continue

        heat = pivot_metric(summary_df, metric)
        outfile = os.path.join(outdir, f"heatmap_{metric}.png")
        draw_simple_heatmap(
            data=heat,
            title=f"{metric} across latent dimensions and traversal deltas",
            xlabel="Delta",
            ylabel="Latent dimension",
            outfile=outfile,
            annotate=False,
            figsize=(8, max(8, heat.shape[0] * 0.28)),
        )
        print(f"Saved: {outfile}")
# Plot 3: motif heatmap

def plot_motif_heatmaps(summary_df: pd.DataFrame, outdir: str):
    motif_metrics = get_available_motif_metrics(summary_df, MOTIFS)
    if not motif_metrics:
        print("[WARN] No motif metrics found in summary.csv")
        return

    for metric in motif_metrics:
        heat = pivot_metric(summary_df, metric)
        outfile = os.path.join(outdir, f"heatmap_{metric}.png")
        draw_simple_heatmap(
            data=heat,
            title=f"{metric} across latent dimensions and traversal deltas",
            xlabel="Delta",
            ylabel="Latent dimension",
            outfile=outfile,
            annotate=False,
            figsize=(8, max(8, heat.shape[0] * 0.28)),
        )
        print(f"Saved: {outfile}")

    rows = []
    for metric in motif_metrics:
        try:
            contrast = compute_pos_neg_contrast(summary_df, metric, pos_delta=3.0, neg_delta=-3.0)
            rows.append(contrast.rename(metric))
        except ValueError:
            continue

    if rows:
        combined = pd.concat(rows, axis=1).T
        outfile = os.path.join(outdir, "motif_contrast_heatmap_pos3_minus_neg3.png")
        draw_simple_heatmap(
            data=combined,
            title="Motif presence contrast (+3 minus -3)",
            xlabel="Latent dimension",
            ylabel="Motif metric",
            outfile=outfile,
            annotate=False,
            figsize=(max(9, combined.shape[1] * 0.35), max(4, combined.shape[0] * 0.7)),
        )
        print(f"Saved: {outfile}")

def pick_representative_dimensions(summary_df: pd.DataFrame) -> Dict[str, List[int]]:

    property_to_metric = {
        "length": "mean_length",
        "aromatic": "mean_group_aromatic",
        "positive": "mean_group_positive",
        "glycine": "mean_group_glycine",
    }

    selected = {}
    for prop_name, metric in property_to_metric.items():
        if metric not in summary_df.columns:
            continue
        try:
            contrast = compute_pos_neg_contrast(summary_df, metric, pos_delta=3.0, neg_delta=-3.0)
        except ValueError:
            continue

        dims = contrast.abs().sort_values(ascending=False).head(TOP_DIMS_PER_PROPERTY).index.tolist()
        selected[prop_name] = dims

    return selected


def create_representative_sequence_table(summary_df: pd.DataFrame, results_df: pd.DataFrame, outdir: str):
    selected = pick_representative_dimensions(summary_df)
    baseline = choose_baseline_seq(results_df)

    if baseline is None:
        print("[WARN] No baseline sequence found in results.csv")
        return

    rows = []
    for property_name, dims in selected.items():
        metric_map = {
            "length": "mean_length",
            "aromatic": "mean_group_aromatic",
            "positive": "mean_group_positive",
            "glycine": "mean_group_glycine",
        }
        metric = metric_map[property_name]

        for dim in dims:
            row = {
                "property": property_name,
                "latent_dim": dim,
                "metric": metric,
                "baseline_seq": baseline["sequence"],
                "baseline_len": baseline["pred_len"],
            }

            for d in REP_DELTAS:
                rep = choose_representative_seq(results_df, latent_dim=dim, delta=d)
                if rep is None:
                    row[f"delta_{d}_seq"] = "NA"
                    row[f"delta_{d}_len"] = "NA"
                else:
                    row[f"delta_{d}_seq"] = rep["sequence"]
                    row[f"delta_{d}_len"] = rep["pred_len"]

            for d in REP_DELTAS:
                sub = summary_df[(summary_df["latent_dim"] == dim) & (summary_df["delta"] == d)]
                if sub.empty or metric not in sub.columns:
                    row[f"delta_{d}_{metric}"] = np.nan
                else:
                    row[f"delta_{d}_{metric}"] = float(sub.iloc[0][metric])

            rows.append(row)

    if not rows:
        print("[WARN] No representative rows generated")
        return

    rep_df = pd.DataFrame(rows)
    rep_csv = os.path.join(outdir, "representative_sequences_table.csv")
    rep_df.to_csv(rep_csv, index=False)
    print(f"Saved: {rep_csv}")

    n_rows = len(rep_df) + 1
    n_cols = 6  # property, dim, baseline, -3, 0, +3
    fig_h = max(4, n_rows * 1.15)
    fig_w = 18

    fig, ax = plt.subplots(figsize=(fig_w, fig_h))
    ax.axis("off")

    headers = ["Property", "Latent dim", "Baseline", "Delta -3", "Delta 0", "Delta +3"]
    cell_text = []

    for _, r in rep_df.iterrows():
        property_label = r["property"]
        dim_label = str(r["latent_dim"])

        baseline_text = f"L={r['baseline_len']}\n{wrap_seq(str(r['baseline_seq']), SEQ_WRAP)}"

        d_neg3_seq = str(r.get("delta_-3.0_seq", "NA"))
        d_neg3_len = r.get("delta_-3.0_len", "NA")
        d_neg3_metric = r.get(f"delta_-3.0_{r['metric']}", np.nan)
        d_neg3_text = f"L={d_neg3_len}\n{r['metric']}={d_neg3_metric:.3f}\n{wrap_seq(d_neg3_seq, SEQ_WRAP)}" \
            if pd.notna(d_neg3_metric) and d_neg3_seq != "NA" else "NA"

        d_zero_seq = str(r.get("delta_0.0_seq", "NA"))
        d_zero_len = r.get("delta_0.0_len", "NA")
        d_zero_metric = r.get(f"delta_0.0_{r['metric']}", np.nan)
        d_zero_text = f"L={d_zero_len}\n{r['metric']}={d_zero_metric:.3f}\n{wrap_seq(d_zero_seq, SEQ_WRAP)}" \
            if pd.notna(d_zero_metric) and d_zero_seq != "NA" else "NA"

        d_pos3_seq = str(r.get("delta_3.0_seq", "NA"))
        d_pos3_len = r.get("delta_3.0_len", "NA")
        d_pos3_metric = r.get(f"delta_3.0_{r['metric']}", np.nan)
        d_pos3_text = f"L={d_pos3_len}\n{r['metric']}={d_pos3_metric:.3f}\n{wrap_seq(d_pos3_seq, SEQ_WRAP)}" \
            if pd.notna(d_pos3_metric) and d_pos3_seq != "NA" else "NA"

        cell_text.append([
            property_label,
            dim_label,
            baseline_text,
            d_neg3_text,
            d_zero_text,
            d_pos3_text,
        ])

    table = ax.table(
        cellText=cell_text,
        colLabels=headers,
        loc="center",
        cellLoc="left",
        colLoc="center",
    )

    table.auto_set_font_size(False)
    table.set_fontsize(8)
    table.scale(1.0, 2.2)

    col_widths = {
        0: 0.08,
        1: 0.07,
        2: 0.22,
        3: 0.21,
        4: 0.21,
        5: 0.21,
    }

    for (row, col), cell in table.get_celld().items():
        if col in col_widths:
            cell.set_width(col_widths[col])

        if row == 0:
            cell.set_text_props(weight="bold")
            cell.set_height(0.08)
        else:
            cell.set_height(0.16)

    plt.title("Representative sequences under latent traversal", pad=20)
    plt.tight_layout()

    out_png = os.path.join(outdir, "representative_sequences_table.png")
    plt.savefig(out_png, dpi=DPI, bbox_inches="tight")
    plt.close()
    print(f"Saved: {out_png}")

# Extra combined summary plots

def plot_property_contrast_heatmap(summary_df: pd.DataFrame, outdir: str):
    rows = []
    kept_metrics = []
    for metric in PROPERTY_METRICS:
        if metric not in summary_df.columns:
            continue
        try:
            contrast = compute_pos_neg_contrast(summary_df, metric, pos_delta=3.0, neg_delta=-3.0)
        except ValueError:
            continue
        rows.append(contrast.rename(metric))
        kept_metrics.append(metric)

    if not rows:
        print("[WARN] No property contrast heatmap generated")
        return

    combined = pd.concat(rows, axis=1).T
    outfile = os.path.join(outdir, "property_contrast_heatmap_pos3_minus_neg3.png")
    draw_simple_heatmap(
        data=combined,
        title="Property contrast (+3 minus -3)",
        xlabel="Latent dimension",
        ylabel="Property metric",
        outfile=outfile,
        annotate=False,
        figsize=(max(9, combined.shape[1] * 0.35), max(4, combined.shape[0] * 0.75)),
    )
    print(f"Saved: {outfile}")

# Main

def main():
    ensure_outdir(OUTDIR)
    summary_df, results_df = load_data(SUMMARY_CSV, RESULTS_CSV)

    print("Loaded summary:", summary_df.shape)
    print("Loaded results:", results_df.shape)

    plot_length_lines(summary_df, OUTDIR)
    plot_property_heatmaps(summary_df, OUTDIR)
    plot_property_contrast_heatmap(summary_df, OUTDIR)
    plot_motif_heatmaps(summary_df, OUTDIR)
    create_representative_sequence_table(summary_df, results_df, OUTDIR)

    print("\nDone. Plots saved under:", OUTDIR)


if __name__ == "__main__":
    main()
