import re
from dataclasses import dataclass, asdict
from typing import Dict, List, Tuple

import numpy as np
import pandas as pd

AMINO_ACIDS = set("ACDEFGHIKLMNPQRSTVWY")
KD_SCALE = {
    "I": 4.5, "V": 4.2, "L": 3.8, "F": 2.8, "C": 2.5, "M": 1.9, "A": 1.8,
    "G": -0.4, "T": -0.7, "S": -0.8, "W": -0.9, "Y": -1.3, "P": -1.6,
    "H": -3.2, "E": -3.5, "Q": -3.5, "D": -3.5, "N": -3.5, "K": -3.9, "R": -4.5
}
PKA = {
    "Cterm": 3.1, "Nterm": 8.0,
    "C": 8.5, "D": 3.9, "E": 4.1, "H": 6.0, "K": 10.5, "R": 12.5, "Y": 10.1
}
POSITIVE = set("KRH")
NEGATIVE = set("DE")
HYDROPHOBIC = set("AILMFWVYC")
AROMATIC = set("FWY")

ABS_THRESHOLDS = {
    "cdr3_len_min": 8,
    "cdr3_len_max": 20,
    "extra_cys_heavy_max": 0,
    "extra_cys_cdr3_max": 0,
    "n_glyco_motif_heavy_max": 0,
    "n_glyco_motif_cdr3_max": 0,
    "deamidation_motif_cdr3_max": 1,
    "isomerization_motif_cdr3_max": 1,
    "max_hydrophobic_run_cdr3_max": 5,
    "gravy_cdr3_max": 0.8,
    "net_charge_cdr3_abs_max": 4,
    "oxidation_m_count_cdr3_max": 1,
}


def clean_seq(seq: str) -> str:
    seq = str(seq).strip().upper()
    return "".join(ch for ch in seq if ch in AMINO_ACIDS)


def clean_target(x: str) -> str:
    return str(x).strip()


def levenshtein(s1: str, s2: str) -> int:
    if s1 == s2:
        return 0
    if len(s1) < len(s2):
        s1, s2 = s2, s1
    previous = list(range(len(s2) + 1))
    for i, c1 in enumerate(s1, start=1):
        current = [i]
        for j, c2 in enumerate(s2, start=1):
            ins = previous[j] + 1
            dele = current[j - 1] + 1
            sub = previous[j - 1] + (c1 != c2)
            current.append(min(ins, dele, sub))
        previous = current
    return previous[-1]


def normalized_edit_distance(s1: str, s2: str) -> float:
    denom = max(len(s1), len(s2), 1)
    return levenshtein(s1, s2) / denom


def hydrophobic_run_length(seq: str) -> int:
    longest = cur = 0
    for aa in seq:
        if aa in HYDROPHOBIC:
            cur += 1
            longest = max(longest, cur)
        else:
            cur = 0
    return longest


def count_motif_regex(seq: str, pattern: str) -> int:
    return len(list(re.finditer(pattern, seq)))


def count_deamidation_motifs(seq: str) -> int:
    return seq.count("NG") + seq.count("NS")


def count_isomerization_motifs(seq: str) -> int:
    return sum(seq.count(m) for m in ("DG", "DS", "DT", "DD"))


def count_oxidation_hotspots(seq: str) -> int:
    return seq.count("M")


def count_extra_cys(seq: str, expected_min: int = 2) -> int:
    c = seq.count("C")
    return max(0, c - expected_min)


def gravy(seq: str) -> float:
    if not seq:
        return np.nan
    return float(np.mean([KD_SCALE[a] for a in seq]))


def net_charge_at_ph(seq: str, ph: float = 7.4) -> float:
    if not seq:
        return np.nan

    def pos_fraction(pka_val: float) -> float:
        return 1.0 / (1.0 + 10 ** (ph - pka_val))

    def neg_fraction(pka_val: float) -> float:
        return 1.0 / (1.0 + 10 ** (pka_val - ph))

    charge = 0.0
    charge += pos_fraction(PKA["Nterm"])
    charge -= neg_fraction(PKA["Cterm"])

    for aa in seq:
        if aa == "K":
            charge += pos_fraction(PKA["K"])
        elif aa == "R":
            charge += pos_fraction(PKA["R"])
        elif aa == "H":
            charge += pos_fraction(PKA["H"])
        elif aa == "D":
            charge -= neg_fraction(PKA["D"])
        elif aa == "E":
            charge -= neg_fraction(PKA["E"])
        elif aa == "C":
            charge -= neg_fraction(PKA["C"])
        elif aa == "Y":
            charge -= neg_fraction(PKA["Y"])
    return float(charge)


def estimate_pI(seq: str) -> float:
    lo, hi = 0.0, 14.0
    for _ in range(60):
        mid = 0.5 * (lo + hi)
        charge = net_charge_at_ph(seq, ph=mid)
        if charge > 0:
            lo = mid
        else:
            hi = mid
    return float(0.5 * (lo + hi))


def rolling_window_max(seq: str, func, window: int = 5) -> float:
    if len(seq) == 0:
        return np.nan
    if len(seq) <= window:
        return func(seq)
    vals = [func(seq[i:i + window]) for i in range(len(seq) - window + 1)]
    return float(np.max(vals))


def robust_zscore(series: pd.Series) -> pd.Series:
    med = series.median()
    mad = np.median(np.abs(series - med))
    if mad < 1e-12:
        return pd.Series(np.zeros(len(series)), index=series.index)
    return 0.6745 * (series - med) / mad


@dataclass
class DevFeatures:
    heavy_len: int
    cdr3_len: int
    heavy_gravy: float
    cdr3_gravy: float
    heavy_net_charge_pH74: float
    cdr3_net_charge_pH74: float
    heavy_pI: float
    cdr3_pI: float
    heavy_hydrophobic_run: int
    cdr3_hydrophobic_run: int
    heavy_frac_hydrophobic: float
    cdr3_frac_hydrophobic: float
    heavy_frac_aromatic: float
    cdr3_frac_aromatic: float
    heavy_frac_positive: float
    cdr3_frac_positive: float
    heavy_frac_negative: float
    cdr3_frac_negative: float
    heavy_n_glyco_motifs: int
    cdr3_n_glyco_motifs: int
    heavy_deamidation_motifs: int
    cdr3_deamidation_motifs: int
    heavy_isomerization_motifs: int
    cdr3_isomerization_motifs: int
    heavy_oxidation_m_count: int
    cdr3_oxidation_m_count: int
    heavy_extra_cys_proxy: int
    cdr3_extra_cys_proxy: int
    cdr3_max_window_gravy_5: float
    cdr3_max_window_abs_charge_5: float


def compute_features(heavy: str, cdr3: str) -> DevFeatures:
    heavy = clean_seq(heavy)
    cdr3 = clean_seq(cdr3)

    def frac(seq: str, aa_set: set) -> float:
        return sum(aa in aa_set for aa in seq) / max(len(seq), 1)

    def abs_charge_local(seq: str) -> float:
        return abs(net_charge_at_ph(seq, ph=7.4))

    return DevFeatures(
        heavy_len=len(heavy),
        cdr3_len=len(cdr3),
        heavy_gravy=gravy(heavy),
        cdr3_gravy=gravy(cdr3),
        heavy_net_charge_pH74=net_charge_at_ph(heavy, 7.4),
        cdr3_net_charge_pH74=net_charge_at_ph(cdr3, 7.4),
        heavy_pI=estimate_pI(heavy),
        cdr3_pI=estimate_pI(cdr3),
        heavy_hydrophobic_run=hydrophobic_run_length(heavy),
        cdr3_hydrophobic_run=hydrophobic_run_length(cdr3),
        heavy_frac_hydrophobic=frac(heavy, HYDROPHOBIC),
        cdr3_frac_hydrophobic=frac(cdr3, HYDROPHOBIC),
        heavy_frac_aromatic=frac(heavy, AROMATIC),
        cdr3_frac_aromatic=frac(cdr3, AROMATIC),
        heavy_frac_positive=frac(heavy, POSITIVE),
        cdr3_frac_positive=frac(cdr3, POSITIVE),
        heavy_frac_negative=frac(heavy, NEGATIVE),
        cdr3_frac_negative=frac(cdr3, NEGATIVE),
        heavy_n_glyco_motifs=count_motif_regex(heavy, r"N[^P][ST]"),
        cdr3_n_glyco_motifs=count_motif_regex(cdr3, r"N[^P][ST]"),
        heavy_deamidation_motifs=count_deamidation_motifs(heavy),
        cdr3_deamidation_motifs=count_deamidation_motifs(cdr3),
        heavy_isomerization_motifs=count_isomerization_motifs(heavy),
        cdr3_isomerization_motifs=count_isomerization_motifs(cdr3),
        heavy_oxidation_m_count=count_oxidation_hotspots(heavy),
        cdr3_oxidation_m_count=count_oxidation_hotspots(cdr3),
        heavy_extra_cys_proxy=count_extra_cys(heavy, expected_min=2),
        cdr3_extra_cys_proxy=max(0, cdr3.count("C")),
        cdr3_max_window_gravy_5=rolling_window_max(cdr3, gravy, window=5),
        cdr3_max_window_abs_charge_5=rolling_window_max(cdr3, abs_charge_local, window=5),
    )


def features_to_dict(f: DevFeatures) -> Dict[str, float]:
    return asdict(f)


def hard_filter_rule_row(row: pd.Series) -> Tuple[bool, List[str]]:
    reasons = []

    if not (ABS_THRESHOLDS["cdr3_len_min"] <= row["cdr3_len"] <= ABS_THRESHOLDS["cdr3_len_max"]):
        reasons.append(f"CDRH3 length outside [{ABS_THRESHOLDS['cdr3_len_min']}, {ABS_THRESHOLDS['cdr3_len_max']}]")
    if row["heavy_extra_cys_proxy"] > ABS_THRESHOLDS["extra_cys_heavy_max"]:
        reasons.append("extra cysteine(s) in heavy chain")
    if row["cdr3_extra_cys_proxy"] > ABS_THRESHOLDS["extra_cys_cdr3_max"]:
        reasons.append("cysteine present in CDRH3")
    if row["heavy_n_glyco_motifs"] > ABS_THRESHOLDS["n_glyco_motif_heavy_max"]:
        reasons.append("N-linked glycosylation motif in heavy chain")
    if row["cdr3_n_glyco_motifs"] > ABS_THRESHOLDS["n_glyco_motif_cdr3_max"]:
        reasons.append("N-linked glycosylation motif in CDRH3")
    if row["cdr3_deamidation_motifs"] > ABS_THRESHOLDS["deamidation_motif_cdr3_max"]:
        reasons.append("excess deamidation motif(s) in CDRH3")
    if row["cdr3_isomerization_motifs"] > ABS_THRESHOLDS["isomerization_motif_cdr3_max"]:
        reasons.append("excess Asp isomerization motif(s) in CDRH3")
    if row["cdr3_hydrophobic_run"] > ABS_THRESHOLDS["max_hydrophobic_run_cdr3_max"]:
        reasons.append("long hydrophobic run in CDRH3")
    if row["cdr3_gravy"] > ABS_THRESHOLDS["gravy_cdr3_max"]:
        reasons.append("high average hydrophobicity in CDRH3")
    if abs(row["cdr3_net_charge_pH74"]) > ABS_THRESHOLDS["net_charge_cdr3_abs_max"]:
        reasons.append("extreme net charge in CDRH3")
    if row["cdr3_oxidation_m_count"] > ABS_THRESHOLDS["oxidation_m_count_cdr3_max"]:
        reasons.append("too many methionine oxidation hotspots in CDRH3")

    return len(reasons) == 0, reasons


def relative_risk_score(df: pd.DataFrame) -> pd.Series:
    outlier_metrics = [
        "cdr3_gravy",
        "cdr3_hydrophobic_run",
        "cdr3_max_window_gravy_5",
        "cdr3_max_window_abs_charge_5",
        "cdr3_len",
        "heavy_n_glyco_motifs",
        "cdr3_n_glyco_motifs",
        "cdr3_deamidation_motifs",
        "cdr3_isomerization_motifs",
        "heavy_extra_cys_proxy",
        "cdr3_extra_cys_proxy",
    ]

    score = pd.Series(np.zeros(len(df)), index=df.index, dtype=float)
    score += 3.0 * df["heavy_n_glyco_motifs"]
    score += 4.0 * df["cdr3_n_glyco_motifs"]
    score += 2.0 * df["cdr3_deamidation_motifs"]
    score += 2.0 * df["cdr3_isomerization_motifs"]
    score += 3.0 * df["heavy_extra_cys_proxy"]
    score += 4.0 * df["cdr3_extra_cys_proxy"]
    score += 0.75 * df["cdr3_oxidation_m_count"]

    for m in outlier_metrics:
        rz = robust_zscore(df[m])
        score += np.maximum(rz, 0.0).astype(float)

    charge_pen = np.maximum(np.abs(df["cdr3_net_charge_pH74"]) - 2.5, 0.0)
    score += 0.75 * charge_pen
    return score


def nearest_neighbor_distance(query_seq: str, ref_seqs: List[str], exclude_identical: bool = True) -> float:
    best = None
    for s in ref_seqs:
        if exclude_identical and query_seq == s:
            continue
        d = normalized_edit_distance(query_seq, s)
        if best is None or d < best:
            best = d
    return np.nan if best is None else float(best)
