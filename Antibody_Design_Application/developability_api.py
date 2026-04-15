import matplotlib.pyplot as plt
from pathlib import Path
from typing import List, Dict

import pandas as pd

from developability_model import (
    clean_seq,
    clean_target,
    compute_features,
    features_to_dict,
    hard_filter_rule_row,
    relative_risk_score,
    nearest_neighbor_distance,
)


class DevelopabilityRanker:
    def __init__(self, csv_path: str):
        self.csv_path = csv_path
        self.df = pd.read_csv(csv_path)

        required = ["Target", "Heavy", "cdr3"]
        missing = [c for c in required if c not in self.df.columns]
        if missing:
            raise ValueError(f"Missing required columns in {csv_path}: {missing}")

        self.df = self.df[required].dropna().copy()
        self.df["Target"] = self.df["Target"].map(clean_target)
        self.df["Heavy"] = self.df["Heavy"].map(clean_seq)
        self.df["cdr3"] = self.df["cdr3"].map(clean_seq)

        self.df = self.df[
            (self.df["Target"].str.len() > 0) &
            (self.df["Heavy"].str.len() > 0) &
            (self.df["cdr3"].str.len() > 0)
        ].drop_duplicates(subset=["Target", "Heavy", "cdr3"]).reset_index(drop=True)

    def list_targets(self) -> List[str]:
        return sorted(self.df["Target"].dropna().unique().tolist())

    def get_target_cohort(self, target_name: str) -> pd.DataFrame:
        target_name = clean_target(target_name)
        cohort = self.df[self.df["Target"] == target_name].copy()
        if len(cohort) == 0:
            raise ValueError(f"No cohort found for Target='{target_name}'")
        return cohort.reset_index(drop=True)

    def attach_features(self, df: pd.DataFrame) -> pd.DataFrame:
        feats = [features_to_dict(compute_features(h, c)) for h, c in zip(df["Heavy"], df["cdr3"])]
        feat_df = pd.DataFrame(feats)
        out = pd.concat([df.reset_index(drop=True), feat_df.reset_index(drop=True)], axis=1)

        passed_list = []
        reasons_list = []
        for _, row in out.iterrows():
            passed, reasons = hard_filter_rule_row(row)
            passed_list.append(passed)
            reasons_list.append("; ".join(reasons) if reasons else "")
        out["hard_filter_pass"] = passed_list
        out["hard_filter_reasons"] = reasons_list
        return out

    def prepare_selected(self, candidates: List[Dict]) -> pd.DataFrame:
        rows = []
        for item in candidates:
            heavy = clean_seq(item["Heavy"])
            cdr3 = clean_seq(item["cdr3"])
            row = {
                "Target": clean_target(item["Target"]),
                "Heavy": heavy,
                "cdr3": cdr3,
                "candidate_name": item.get("candidate_name", ""),
            }
            row.update(features_to_dict(compute_features(heavy, cdr3)))

            passed, reasons = hard_filter_rule_row(pd.Series(row))
            row["hard_filter_pass"] = passed
            row["hard_filter_reasons"] = "; ".join(reasons) if reasons else ""
            rows.append(row)

        return pd.DataFrame(rows)
    def plot_risk_distribution(
        self,
        target_name: str,
        scored_df: pd.DataFrame,
        out_path: str | None = None,
    ):
        cohort_df = self.get_target_cohort(target_name)
        cohort_df = self.attach_features(cohort_df)
        cohort_df["developability_risk_score"] = relative_risk_score(cohort_df)

        fig, ax = plt.subplots(figsize=(5.0, 3.6))

        vals = cohort_df["developability_risk_score"].dropna().values
        bins = min(28, max(8, len(vals) // 5))

        ax.hist(
            vals,
            bins=bins,
            color="#DDE7F0",
            edgecolor="#7C93AC",
            linewidth=0.8,
            alpha=0.95,
        )

        candidate_colors = ["#3C5488", "#00A087", "#E64B35"]

        for i, (_, row) in enumerate(scored_df.reset_index(drop=True).iterrows()):
            x = row["developability_risk_score"]
            c = candidate_colors[i % len(candidate_colors)]
            ax.axvline(x, color=c, linewidth=0.9, alpha=0.95)

  
        for i, (_, row) in enumerate(scored_df.reset_index(drop=True).iterrows()):
            c = candidate_colors[i % len(candidate_colors)]
            label = row.get("candidate_name", f"C{i+1}")
            ax.text(
                0.98,
                0.98 - i * 0.08,
                label,
                transform=ax.transAxes,
                ha="right",
                va="top",
                fontsize=8,
                color=c,
                fontweight="bold",
            )

        ax.set_xlabel("Developability risk score")
        ax.set_ylabel("Count")
        ax.set_title(f"Risk distribution for {target_name}")

        ax.spines["top"].set_visible(False)
        ax.spines["right"].set_visible(False)
        ax.grid(axis="y", color="#D9D9D9", linewidth=0.6, alpha=0.8)
        ax.set_axisbelow(True)

        fig.tight_layout()

        if out_path is not None:
            out_path = Path(out_path)
            out_path.parent.mkdir(parents=True, exist_ok=True)
            fig.savefig(out_path, dpi=300, bbox_inches="tight", facecolor="white")

        return fig
    def score_candidates(self, target_name: str, candidates: List[Dict]) -> pd.DataFrame:
        cohort_df = self.get_target_cohort(target_name)
        cohort_df = self.attach_features(cohort_df)

        selected_df = self.prepare_selected(candidates)

        combined = pd.concat(
            [cohort_df.assign(_source="cohort"), selected_df.assign(_source="selected")],
            ignore_index=True,
            sort=False,
        )
        combined["developability_risk_score"] = relative_risk_score(combined)

        selected_df = combined[combined["_source"] == "selected"].drop(columns=["_source"]).reset_index(drop=True)

        risk_vals = cohort_df.copy()
        risk_vals["developability_risk_score"] = relative_risk_score(cohort_df)
        risk_ref = risk_vals["developability_risk_score"].dropna().values

        heavy_refs = cohort_df["Heavy"].tolist()
        cdr3_refs = cohort_df["cdr3"].tolist()

        selected_df["heavy_nn_edit_distance"] = selected_df["Heavy"].map(
            lambda x: nearest_neighbor_distance(x, heavy_refs, exclude_identical=True)
        )
        selected_df["cdr3_nn_edit_distance"] = selected_df["cdr3"].map(
            lambda x: nearest_neighbor_distance(x, cdr3_refs, exclude_identical=True)
        )

        selected_df["developability_risk_score_percentile"] = selected_df["developability_risk_score"].map(
            lambda x: 100.0 * (risk_ref <= x).mean() if len(risk_ref) else None
        )

        selected_df["low_risk_claim"] = (
            selected_df["hard_filter_pass"] &
            (selected_df["developability_risk_score_percentile"] <= 50.0)
        )
        selected_df["high_diversity_claim"] = (
            (selected_df["heavy_nn_edit_distance"] >= 0.10) |
            (selected_df["cdr3_nn_edit_distance"] >= 0.20)
        )
        selected_df["overall_claim"] = (
            selected_df["low_risk_claim"] & selected_df["high_diversity_claim"]
        )

        selected_df = selected_df.sort_values(
            by=["overall_claim", "low_risk_claim", "developability_risk_score", "cdr3_nn_edit_distance"],
            ascending=[False, False, True, False],
        ).reset_index(drop=True)

        return selected_df
