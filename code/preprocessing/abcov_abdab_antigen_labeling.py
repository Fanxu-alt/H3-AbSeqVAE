import pandas as pd
import re

input_file = "CoV-AbDab_080224.csv"
output_file = "abCoV-AbDab_processed.csv"

df = pd.read_csv(input_file)

#    A C D E F G H I K L M N P Q R S T V W Y

STANDARD_AA = set("ACDEFGHIKLMNPQRSTVWY")

def is_valid_vh_or_vhh(seq):

    if pd.isna(seq):
        return False

    seq = str(seq).strip().upper()

    if len(seq) <= 4:
        return False

    return all(ch in STANDARD_AA for ch in seq)

def is_non_empty(x):
    if pd.isna(x):
        return False
    return str(x).strip() != ""

def split_antigens(cell):
    if pd.isna(cell):
        return []

    text = str(cell).strip()
    if text == "":
        return []

    parts = re.split(r"\s*;\s*", text)

    parts = [p.strip() for p in parts if p.strip() != ""]

    return parts

filtered_df = df.copy()

filtered_df = filtered_df[filtered_df["Ab or Nb"] == "Ab"]

filtered_df = filtered_df[filtered_df["VHorVHH"].apply(is_valid_vh_or_vhh)]

filtered_df = filtered_df[
    filtered_df["Neutralising Vs"].apply(is_non_empty) &
    filtered_df["Not Neutralising Vs"].apply(is_non_empty)
]

new_rows = []

for _, row in filtered_df.iterrows():
    vh_seq = str(row["VHorVHH"]).strip().upper()
    protein_epitope = row["Protein + Epitope"]

    neutralising_list = split_antigens(row["Neutralising Vs"])
    not_neutralising_list = split_antigens(row["Not Neutralising Vs"])

    for antigen in neutralising_list:
        new_rows.append({
            "VHorVHH": vh_seq,
            "Protein + Epitope": protein_epitope,
            "Antigen": antigen,
            "Label": 1
        })

    for antigen in not_neutralising_list:
        new_rows.append({
            "VHorVHH": vh_seq,
            "Protein + Epitope": protein_epitope,
            "Antigen": antigen,
            "Label": 0
        })

result_df = pd.DataFrame(new_rows)

result_df = result_df.drop_duplicates().reset_index(drop=True)

result_df.to_csv(output_file, index=False, encoding="utf-8-sig")

print("处理完成")
print(f"原始数据行数: {len(df)}")
print(f"过滤后行数: {len(filtered_df)}")
print(f"拆分后结果行数: {len(result_df)}")
print(f"结果文件已保存为: {output_file}")

print("\n结果预览:")
print(result_df.head(10))
