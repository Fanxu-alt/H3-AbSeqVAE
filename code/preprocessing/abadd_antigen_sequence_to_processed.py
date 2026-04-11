import pandas as pd

processed_file = "abCoV-AbDab_processed.csv"
reference_file = "CoV-AbDab_only_sars2_filter.csv"
output_file = "abCoV-AbDab_final_no_empty_seq.csv"

processed_df = pd.read_csv(processed_file)
reference_df = pd.read_csv(reference_file)

def normalize_text(x):
    if pd.isna(x):
        return ""
    return str(x).strip()

processed_df["Antigen_key"] = processed_df["Antigen"].apply(normalize_text)
processed_df["Protein_Epitope_key"] = processed_df["Protein + Epitope"].apply(normalize_text)

reference_df["Target_key"] = reference_df["Target"].apply(normalize_text)
reference_df["Protein_Epitope_key"] = reference_df["Protein_Epitope"].apply(normalize_text)
reference_df["Antigen Sequence"] = reference_df["antigen"].apply(normalize_text)

ref_df = reference_df[
    ["Target_key", "Protein_Epitope_key", "Antigen Sequence"]
].drop_duplicates()

merged_df = processed_df.merge(
    ref_df,
    how="left",
    left_on=["Antigen_key", "Protein_Epitope_key"],
    right_on=["Target_key", "Protein_Epitope_key"]
)

merged_df = merged_df[
    merged_df["Antigen Sequence"].notna() &
    (merged_df["Antigen Sequence"].str.strip() != "")
]

drop_cols = ["Antigen_key", "Protein_Epitope_key", "Target_key"]
drop_cols = [c for c in drop_cols if c in merged_df.columns]

merged_df = merged_df.drop(columns=drop_cols)

merged_df.to_csv(output_file, index=False, encoding="utf-8-sig")

print("处理完成")
print(f"原 processed 行数: {len(processed_df)}")
print(f"合并后行数: {len(merged_df)}")
print(f"最终无空序列行数: {len(merged_df)}")
print(f"结果文件: {output_file}")

print("\n预览：")
print(merged_df.head())
