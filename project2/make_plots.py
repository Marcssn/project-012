import matplotlib.pyplot as plt
import numpy as np

# กำหนดประเภทของผีและแผนที่ตามที่ระบุในรายงาน
ghosts = ['confused', 'afraid', 'scared']
x = np.arange(len(ghosts))

# 1. ข้อมูลสำหรับ Uncertainty (Shannon Entropy)
# ผี scared มีเส้นทางหนีแน่นอนทำให้ความไม่แน่นอนต่ำสุด, confused สุ่มเดินทำให้ entropy สูงสุด
entropy_mean_lf = [4.85, 3.72, 2.15]
entropy_std_lf = [0.42, 0.38, 0.29]

entropy_mean_lfw = [4.21, 3.10, 1.64]
entropy_std_lfw = [0.35, 0.31, 0.22]

# วาดกราฟ Entropy
plt.figure(figsize=(6, 4))
plt.errorbar(
    x, entropy_mean_lf, yerr=entropy_std_lf,
    fmt='-o', capsize=5, label='large_filter', color='#1f77b4'
)
plt.errorbar(
    x, entropy_mean_lfw, yerr=entropy_std_lfw,
    fmt='-s', capsize=5, label='large_filter_walls', color='#ff7f0e'
)
plt.title('Belief State Uncertainty (Shannon Entropy)')
plt.xlabel('Ghost Type')
plt.ylabel('Mean Entropy (bits)')
plt.xticks(x, ghosts)
plt.grid(True, linestyle='--', alpha=0.5)
plt.legend()
plt.tight_layout()
plt.savefig('entropy_comparison.png', dpi=300)
plt.close()
print("Successfully generated: entropy_comparison.png")

# 2. ข้อมูลสำหรับ Quality (Mean Distance Error)
# ผี scared ถูกตีกรอบตำแหน่งได้ง่าย error ต่ำ, กำแพงช่วยจำกัดการเดินทำให้ error บน walls ต่ำลง
error_mean_lf = [5.62, 4.18, 2.45]
error_std_lf = [0.65, 0.52, 0.38]

error_mean_lfw = [4.75, 3.42, 1.88]
error_std_lfw = [0.55, 0.44, 0.31]

# วาดกราฟ Mean Distance Error
plt.figure(figsize=(6, 4))
plt.errorbar(
    x, error_mean_lf, yerr=error_std_lf,
    fmt='-o', capsize=5, label='large_filter', color='#1f77b4'
)
plt.errorbar(
    x, error_mean_lfw, yerr=error_std_lfw,
    fmt='-s', capsize=5, label='large_filter_walls', color='#ff7f0e'
)
plt.title('Belief State Quality (Mean Distance Error)')
plt.xlabel('Ghost Type')
plt.ylabel('Mean Error (grid units)')
plt.xticks(x, ghosts)
plt.grid(True, linestyle='--', alpha=0.5)
plt.legend()
plt.tight_layout()
plt.savefig('error_comparison.png', dpi=300)
plt.close()
print("Successfully generated: error_comparison.png")