import matplotlib.pyplot as plt

# Income distribution
labels = ['High income (>50K)', 'Low income (<=50K)']
sizes = [18568, 280717]

fig, ax = plt.subplots(figsize=(7, 7))

fig.patch.set_alpha(0)
ax.patch.set_alpha(0)

ax.pie(
    sizes,
    labels=labels,
    autopct='%1.2f%%',
    startangle=90,
    counterclock=False,
    # wedgeprops={'edgecolor': 'white', 'linewidth': 1},
    textprops={'fontsize': 12, 'color': 'white'}
)

ax.set_title(
    'Income Class Distribution',
    fontsize=15,
    fontweight='bold',
    color='black'
)

ax.axis('equal')

plt.savefig(
    'income_distribution.png',
    dpi=300,
    bbox_inches='tight',
    transparent=True
)

plt.show()