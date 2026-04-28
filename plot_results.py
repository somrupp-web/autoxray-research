"""Generate AUC-vs-iteration chart from results.tsv."""
import csv, sys
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches

tsv = sys.argv[1] if len(sys.argv) > 1 else 'results.tsv'
out = sys.argv[2] if len(sys.argv) > 2 else 'results_chart.png'

rows, labels, aucs, colors = [], [], [], []
with open(tsv) as f:
    for i, row in enumerate(csv.DictReader(f, delimiter='\t')):
        try:
            auc = float(row['val_auc'])
        except (ValueError, KeyError):
            continue
        rows.append(i)
        labels.append(row.get('description', f'exp-{i}')[:30])
        aucs.append(auc)
        colors.append('#2ecc71' if row.get('status') == 'keep' else '#e74c3c')

fig, ax = plt.subplots(figsize=(12, 5))
bars = ax.bar(range(len(aucs)), aucs, color=colors, edgecolor='white', linewidth=0.5)
ax.axhline(0.841, color='gold', linewidth=2, linestyle='--', label='CheXNet benchmark (0.841)')
ax.axhline(max(aucs), color='cyan', linewidth=1, linestyle=':', alpha=0.8, label=f'Current best ({max(aucs):.4f})')
ax.set_xticks(range(len(labels)))
ax.set_xticklabels(labels, rotation=35, ha='right', fontsize=7)
ax.set_ylabel('val AUC-ROC')
ax.set_title('Autonomous X-ray Research — OpenCode + Qwen3.5-122B on DGX Spark')
ax.set_ylim(0.5, 0.90)
keep_patch = mpatches.Patch(color='#2ecc71', label='Keep (improvement)')
disc_patch = mpatches.Patch(color='#e74c3c', label='Discard')
ax.legend(handles=[keep_patch, disc_patch,
          plt.Line2D([0],[0], color='gold', lw=2, ls='--', label='CheXNet 0.841'),
          plt.Line2D([0],[0], color='cyan', lw=1, ls=':', label=f'Best {max(aucs):.4f}')],
          loc='lower right', fontsize=8)
plt.tight_layout()
plt.savefig(out, dpi=150)
print(f'Chart saved to {out}')
