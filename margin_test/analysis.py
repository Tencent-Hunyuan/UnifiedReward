import pandas as pd
import matplotlib.pyplot as plt
import numpy as np

filter_zero = True

df = pd.read_csv("margin_test\pair_scores.csv")

if filter_zero:
    df = df[(df["score_A"] != 0) & (df["score_B"] != 0)]

df["margin"] = abs(df["score_A"] - df["score_B"])
df["pred"] = (df["score_A"] < df["score_B"]).astype(int)
df["correct"] = (df["pred"] == df["label"]).astype(int)

bin_width = 10
max_margin = df["margin"].max()
bins = np.arange(0, max_margin + bin_width, bin_width)
df["margin_bin"] = pd.cut(df["margin"], bins)

grouped = df.groupby("margin_bin")["correct"]
accuracy_per_bin = grouped.mean()
count_per_bin = grouped.count()

plt.figure(figsize=(10, 6))
accuracy_per_bin.plot(kind="bar", color="skyblue", edgecolor="black")

plt.title("Accuracy vs Margin")
plt.xlabel("Margin Range")
plt.ylabel("Accuracy")
plt.xticks(rotation=45)
plt.grid(axis="y")
plt.tight_layout()

print("Margin区间统计：")
for i, (interval, acc) in enumerate(accuracy_per_bin.items()):
    count = count_per_bin.iloc[i]
    print(f"{interval}: count={count}, accuracy={acc:.2f}")

# plt.savefig("margin_vs_accuracy_filter_zero.png")
plt.show()
