import os
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

sns.set(style="whitegrid")

DATA_DIR = "."
OUT_DIR = "length_head_plots"

os.makedirs(OUT_DIR, exist_ok=True)

#1 Actual Length vs. Predicted Length

df = pd.read_csv(f"{DATA_DIR}/01_true_vs_pred_details.csv")

plt.figure(figsize=(6,6))
plt.scatter(df["true_len"], df["pred_len"], alpha=0.4)

max_len = max(df["true_len"].max(), df["pred_len"].max())

plt.plot([0,max_len],[0,max_len],'r--')

plt.xlabel("True CDRH3 Length")
plt.ylabel("Predicted Length")
plt.title("True vs Predicted CDRH3 Length")

plt.savefig(f"{OUT_DIR}/true_vs_pred_scatter.png",dpi=300)
plt.close()

print("saved scatter plot")

# 2 Confusion Matrix

cm = pd.read_csv(f"{DATA_DIR}/01_confusion_matrix.csv",header=None)

plt.figure(figsize=(8,6))
sns.heatmap(cm,cmap="viridis")

plt.xlabel("Predicted Length")
plt.ylabel("True Length")
plt.title("Length Prediction Confusion Matrix")

plt.savefig(f"{OUT_DIR}/confusion_matrix_heatmap.png",dpi=300)
plt.close()

print("saved confusion matrix")

# 3 Antigen-conditioned length distribution

df = pd.read_csv(f"{DATA_DIR}/02_antigen_conditioned_length_probs.csv")

length_cols = [c for c in df.columns if "pred_prob_len_" in c]

mean_probs = df[length_cols].mean()

lengths = [int(c.split("_")[-1]) for c in length_cols]

plt.figure(figsize=(8,4))
plt.bar(lengths,mean_probs)

plt.xlabel("CDRH3 Length")
plt.ylabel("Mean Probability")
plt.title("Average Predicted Length Distribution")

plt.savefig(f"{OUT_DIR}/length_distribution.png",dpi=300)
plt.close()

print("saved length distribution")

# 4 latent traversal

df = pd.read_csv(f"{DATA_DIR}/03_latent_traversal_summary.csv")

sample = df[df["latent_dim"]<8]

plt.figure(figsize=(8,6))

for dim in sorted(sample["latent_dim"].unique()):
    sub = sample[sample["latent_dim"]==dim]
    plt.plot(sub["delta"],sub["pred_len"],label=f"dim {dim}")

plt.xlabel("Latent Delta")
plt.ylabel("Predicted Length")
plt.title("Latent Traversal Effect on Length")

plt.legend()

plt.savefig(f"{OUT_DIR}/latent_traversal_length.png",dpi=300)
plt.close()

print("saved latent traversal plot")

# 5 latent dimension importance

df = pd.read_csv(f"{DATA_DIR}/04_latent_dim_summary.csv")

df = df.sort_values("mean_length_range",ascending=False)

top = df.head(20)

plt.figure(figsize=(10,5))
plt.bar(top["latent_dim"],top["mean_length_range"])

plt.xlabel("Latent Dimension")
plt.ylabel("Mean Length Range")
plt.title("Latent Dimensions Affecting Length")

plt.savefig(f"{OUT_DIR}/latent_dimension_importance.png",dpi=300)
plt.close()

print("saved latent dimension importance")
