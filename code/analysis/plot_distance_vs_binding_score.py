import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

df = pd.read_csv("cdrh3_distance_vs_score.csv")

print(df.columns)

distance_col = "min_levenshtein_distance"
score_col = "binding_score"

df = df.sort_values(distance_col)

plt.figure(figsize=(10,6))

sns.boxplot(
    data=df,
    x=distance_col,
    y=score_col,
    color="skyblue"
)

sns.stripplot(
    data=df,
    x=distance_col,
    y=score_col,
    color="black",
    alpha=0.4,
    size=3
)

plt.xlabel("Min Levenshtein distance to natural CDRH3")
plt.ylabel("Binding score")
plt.title("Binding score vs CDRH3 distance to natural antibodies")

plt.tight_layout()

plt.savefig("distance_vs_binding_score_boxplot.png", dpi=300)

plt.show()
