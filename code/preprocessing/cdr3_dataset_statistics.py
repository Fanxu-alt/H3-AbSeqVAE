import pandas as pd
from collections import Counter

CSV_PATH = "CoV-AbDab.csv"
COL = "cdr3"

VALID_AA = set("ACDEFGHIKLMNPQRSTVWY")

df = pd.read_csv(CSV_PATH)

cdr3 = (
    df[COL]
    .dropna()
    .astype(str)
    .str.strip()
    .str.upper()
)

# Filter illegal characters
cdr3 = [s for s in cdr3 if all(c in VALID_AA for c in s)]

lengths = [len(s) for s in cdr3]

print("===== BASIC STATS =====")
print("Total sequences:", len(cdr3))
print("Avg length:", sum(lengths) / len(lengths))
print("Median length:", sorted(lengths)[len(lengths)//2])
print("Min length:", min(lengths))
print("Max length:", max(lengths))

print("\n===== LENGTH DISTRIBUTION =====")
length_counts = Counter(lengths)

for l in sorted(length_counts):
    print(f"len={l:2d}  count={length_counts[l]}")

print("\n===== AA FREQUENCY =====")
aa_counter = Counter()

for s in cdr3:
    aa_counter.update(s)

total_aa = sum(aa_counter.values())

for aa, count in sorted(aa_counter.items()):
    print(f"{aa}: {count/total_aa:.4f}")

