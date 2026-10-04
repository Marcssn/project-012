import pandas as pd
import matplotlib.pyplot as plt

# โหลดข้อมูลจาก metrics.csv ที่บันทึกไว้
df = pd.read_csv(
    "metrics.csv",
    names=["layout", "ghost", "entropy", "error"]
)

# คำนวณค่าเฉลี่ยและส่วนเบี่ยงเบนมาตรฐาน (Mean & Std)
grouped = df.groupby(["layout", "ghost"]).agg(
    {"entropy": ["mean", "std"], "error": ["mean", "std"]}
).reset_index()

# 1. กราฟ Entropy (Uncertainty)
fig, ax = plt.subplots(figsize=(7, 4))
for layout in df["layout"].unique():
    sub = grouped[grouped["layout"] == layout]
    ax.errorbar(
        sub["ghost"],
        sub["entropy"]["mean"],
        yerr=sub["entropy"]["std"],
        fmt='-o',
        capsize=5,
        label=layout
    )
ax.set_title("Belief State Uncertainty (Shannon Entropy)")
ax.set_xlabel("Ghost Type")
ax.set_ylabel("Mean Entropy (bits)")
ax.legend()
plt.tight_layout()
plt.savefig("entropy_comparison.png")
plt.close()

# 2. กราฟ Mean Distance Error (Quality)
fig, ax = plt.subplots(figsize=(7, 4))
for layout in df["layout"].unique():
    sub = grouped[grouped["layout"] == layout]
    ax.errorbar(
        sub["ghost"],
        sub["error"]["mean"],
        yerr=sub["error"]["std"],
        fmt='-s',
        capsize=5,
        label=layout
    )
ax.set_title("Belief State Quality (Mean Distance Error)")
ax.set_xlabel("Ghost Type")
ax.set_ylabel("Mean Error (grid units)")
ax.legend()
plt.tight_layout()
plt.savefig("error_comparison.png")
plt.close()

print("Generated entropy_comparison.png and error_comparison.png successfully!")