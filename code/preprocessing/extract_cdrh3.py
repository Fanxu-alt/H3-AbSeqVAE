import pandas as pd
from anarci import anarci

def clean_seq(seq):
    seq = str(seq).strip().upper()
    allowed = set("ACDEFGHIKLMNPQRSTVWY")
    return "".join(ch for ch in seq if ch in allowed)

def extract_cdrh3(seq):
    seq = clean_seq(seq)
    if not seq:
        return None

    try:
        result = anarci([("seq", seq)], scheme="imgt")

        if result is None or result[0] is None or len(result[0]) == 0 or result[0][0] is None:
            return None

        numbering = result[0][0][0][0]

        cdrh3 = []
        for pos, aa in numbering:
            if aa != "-" and 105 <= pos[0] <= 117:
                cdrh3.append(aa)

        return "".join(cdrh3) if cdrh3 else None

    except Exception:
        return None

df = pd.read_csv("heavy.csv")

seq_col = "heavy"
df[seq_col] = df[seq_col].apply(clean_seq)

cdrh3_list = []
for i, seq in enumerate(df[seq_col], 1):
    if i % 100 == 0:
        print(f"Processing {i}/{len(df)} ...")
    cdrh3_list.append(extract_cdrh3(seq))

df["CDRH3"] = cdrh3_list

print("Extracted:", df["CDRH3"].notna().sum(), "/", len(df))
df.to_csv("heavy_with_cdrh3.csv", index=False)
