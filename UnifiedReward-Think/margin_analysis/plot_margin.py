import pandas as pd
import matplotlib.pyplot as plt
import io

stats_file = "margin_analysis/logs/margin_0/stats_after_750.txt"
with open(stats_file, 'r') as f:
    txt = f.read()

df = pd.read_csv(
    io.StringIO(txt),
    sep=r'\s+',
    header=None,
    skiprows=[0, 1],
    names=['margin', 'num_samples', 'accuracy']
)
df.set_index('margin', inplace=True)

# Plotting the data
fig, ax1 = plt.subplots(figsize=(10,5))

# axis 1：num_samples
ax1.bar(df.index, df['num_samples'], color='skyblue', alpha=0.6)
ax1.set_xlabel("Margin")
ax1.set_ylabel("Num Samples", color='mediumblue')
ax1.tick_params(axis='y', labelcolor='mediumblue')

# axis 2: accuracy
ax2 = ax1.twinx()
ax2.plot(df.index, df['accuracy'], color='red', marker='o')
ax2.set_ylabel("Accuracy", color='crimson')
ax2.tick_params(axis='y', labelcolor='crimson')

plt.title("Samples & Accuracy by Margin")
ax1.set_xticks(df.index)
plt.grid(True)
fig.tight_layout()
plt.savefig(stats_file.split('.txt')[0] + '.png')
plt.show()
