import re
import math
import pandas as pd
from typing import List, Tuple, Optional

# Config

GENERATED_TXT = "generated_cdrh3_from_antigenscratch.txt"
TRAIN_CSV = "CoV-AbDab.csv"
TRAIN_CDR3_COL = "cdr3"


SCORER_CSV = None

# SCORER_CSV = "generated_with_scores.csv"
# This document recommends at least two columns：sequence, score

VALID_AA = set("ACDEFGHIKLMNPQRSTVWY")
MIN_LEN = 5
MAX_LEN = 30

# 1. Read the generated sequence

def load_generated_sequences(txt_path: str) -> List[str]:
    seqs = []

    with open(txt_path, "r") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue

            parts = re.split(r"\t+", line)
            if len(parts) >= 3:
                seq = parts[-1].strip().upper()
                seqs.append(seq)
            else:

                continue

    return seqs

# 2. Determine if the sequence is valid

def is_valid_sequence(seq: str, min_len: int = 5, max_len: int = 30) -> bool:
    if not isinstance(seq, str):
        return False
    seq = seq.strip().upper()
    if len(seq) == 0:
        return False
    if len(seq) < min_len or len(seq) > max_len:
        return False
    if any(ch not in VALID_AA for ch in seq):
        return False
    return True

# 3. （Levenshtein）

def levenshtein_distance(s1: str, s2: str) -> int:
    if len(s1) < len(s2):
        s1, s2 = s2, s1

    previous = list(range(len(s2) + 1))
    for i, c1 in enumerate(s1, start=1):
        current = [i]
        for j, c2 in enumerate(s2, start=1):
            insertions = previous[j] + 1
            deletions = current[j - 1] + 1
            substitutions = previous[j - 1] + (c1 != c2)
            current.append(min(insertions, deletions, substitutions))
        previous = current
    return previous[-1]

# 4. novelty

def nearest_neighbor_novelty(seq: str, train_seqs: List[str], normalize: bool = True) -> float:
    best = None
    best_ref_len = None

    for ref in train_seqs:
        d = levenshtein_distance(seq, ref)
        if best is None or d < best:
            best = d
            best_ref_len = max(len(seq), len(ref))

    if best is None:
        return float("nan")

    if normalize:
        return best / best_ref_len if best_ref_len > 0 else 0.0
    return float(best)

# 5. scorer 

def load_scores(score_csv: str) -> pd.DataFrame:
    df = pd.read_csv(score_csv)

    required_cols = {"sequence", "score"}
    if not required_cols.issubset(df.columns):
        raise ValueError(f"{score_csv} must contain columns: {required_cols}")

    df["sequence"] = df["sequence"].astype(str).str.strip().str.upper()
    return df[["sequence", "score"]]

# 6. Main statistical function

def evaluate_sequences(
    generated_seqs: List[str],
    train_seqs: List[str],
    scorer_df: Optional[pd.DataFrame] = None,
):
    total = len(generated_seqs)

    valid_seqs = [s for s in generated_seqs if is_valid_sequence(s, MIN_LEN, MAX_LEN)]
    n_valid = len(valid_seqs)

    valid_pct = 100.0 * n_valid / total if total > 0 else 0.0

    unique_valid = sorted(set(valid_seqs))
    unique_pct = 100.0 * len(unique_valid) / n_valid if n_valid > 0 else 0.0

    avg_length = sum(len(s) for s in valid_seqs) / n_valid if n_valid > 0 else float("nan")

    novelty_list = []
    for s in valid_seqs:
        novelty = nearest_neighbor_novelty(s, train_seqs, normalize=True)
        novelty_list.append(novelty)

    avg_novelty = sum(novelty_list) / len(novelty_list) if novelty_list else float("nan")

    avg_scorer = float("nan")
    top10_scorer = float("nan")

    if scorer_df is not None:
        score_map = dict(zip(scorer_df["sequence"], scorer_df["score"]))

        scores = []
        for s in valid_seqs:
            if s in score_map:
                scores.append(float(score_map[s]))

        if len(scores) > 0:
            avg_scorer = sum(scores) / len(scores)
            top_scores = sorted(scores, reverse=True)[:10]
            top10_scorer = sum(top_scores) / len(top_scores)

    return {
        "total_generated": total,
        "valid_count": n_valid,
        "valid_pct": valid_pct,
        "unique_valid_count": len(unique_valid),
        "unique_pct": unique_pct,
        "avg_length": avg_length,
        "avg_novelty": avg_novelty,
        "avg_scorer": avg_scorer,
        "top10_scorer": top10_scorer,
    }

# 7. Main

def main():
    generated_seqs = load_generated_sequences(GENERATED_TXT)
    print(f"Loaded generated sequences: {len(generated_seqs)}")

    train_df = pd.read_csv(TRAIN_CSV)
    train_seqs = (
        train_df[TRAIN_CDR3_COL]
        .dropna()
        .astype(str)
        .str.strip()
        .str.upper()
        .tolist()
    )
    print(f"Loaded training CDR3 sequences: {len(train_seqs)}")

    scorer_df = None
    if SCORER_CSV is not None:
        scorer_df = load_scores(SCORER_CSV)
        print(f"Loaded scorer file: {len(scorer_df)} rows")

    metrics = evaluate_sequences(generated_seqs, train_seqs, scorer_df)

    print("\n===== Evaluation Metrics =====")
    for k, v in metrics.items():
        print(f"{k}: {v}")


if __name__ == "__main__":
    main()

