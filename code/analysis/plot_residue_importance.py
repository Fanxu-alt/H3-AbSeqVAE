import pandas as pd
import matplotlib.pyplot as plt

heavy = pd.read_csv("heavy_residue_importance.csv")
antigen = pd.read_csv("antigen_residue_importance.csv")

# 1 Residue importance curve

plt.figure(figsize=(10,4))
plt.plot(heavy["position_1based"], heavy["importance"])
plt.xlabel("Heavy chain residue position")
plt.ylabel("Importance")
plt.title("Heavy Chain Residue Importance (Gradient Attribution)")
plt.tight_layout()
plt.savefig("heavy_importance_curve.png", dpi=300)
plt.close()

plt.figure(figsize=(12,4))
plt.plot(antigen["position_1based"], antigen["importance"])
plt.xlabel("Antigen residue position")
plt.ylabel("Importance")
plt.title("Antigen Residue Importance (Gradient Attribution)")
plt.tight_layout()
plt.savefig("antigen_importance_curve.png", dpi=300)
plt.close()

# 2 Top residues bar plot

top_heavy = heavy.sort_values("importance", ascending=False).head(10)

plt.figure(figsize=(6,4))
plt.bar(
    top_heavy["position_1based"].astype(str),
    top_heavy["importance"]
)
plt.xlabel("Residue position")
plt.ylabel("Importance")
plt.title("Top Heavy Chain Residues")
plt.tight_layout()
plt.savefig("heavy_top_residues.png", dpi=300)
plt.close()

top_antigen = antigen.sort_values("importance", ascending=False).head(10)

plt.figure(figsize=(6,4))
plt.bar(
    top_antigen["position_1based"].astype(str),
    top_antigen["importance"]
)
plt.xlabel("Residue position")
plt.ylabel("Importance")
plt.title("Top Antigen Residues")
plt.tight_layout()
plt.savefig("antigen_top_residues.png", dpi=300)
plt.close()

print("Plots saved.")
