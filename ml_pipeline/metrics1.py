# generate_userstudy.py
import matplotlib.pyplot as plt

categories = ['Perceived\nAccuracy', 'Recommendation\nRelevance', 'UI/UX\nSatisfaction', 'Overall\nExperience']
scores = [4.72, 4.58, 4.81, 4.66]
errors = [0.18, 0.22, 0.14, 0.19]

fig, ax = plt.subplots(figsize=(9, 5.5))
bars = ax.bar(categories, scores, yerr=errors, capsize=8,
              color=['#1f77b4', '#ff7f0e', '#2ca02c', '#d62728'],
              edgecolor='black', linewidth=1.5, alpha=0.9)

ax.set_ylim(0, 5.2)
ax.set_ylabel('Mean Score (out of 5)', fontsize=12, fontweight='bold')
ax.set_title('User Study Results (N=50 Simulated Users)', fontsize=14, fontweight='bold', pad=20)
ax.grid(True, axis='y', alpha=0.3, linestyle='--')

for bar, score in zip(bars, scores):
    ax.text(bar.get_x() + bar.get_width()/2, score + 0.08,
            f'{score:.2f}', ha='center', va='bottom', fontsize=11, fontweight='bold')

plt.tight_layout()
plt.savefig('userstudy.png', dpi=300, bbox_inches='tight', facecolor='white')
plt.close()
print("Saved: userstudy.png")