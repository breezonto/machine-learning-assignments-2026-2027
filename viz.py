import numpy as np
import matplotlib.pyplot as plt
from matplotlib.colors import LinearSegmentedColormap, PowerNorm

# --------------------------------------------------
# Feature groups
# --------------------------------------------------

feature_groups = [
    "Numerical",
    "Class of Worker",
    "Industry",
    "Occupation",
    "Education",
    "Marital Status",
    "Major Industry",
    "Major Occupation",
    "Race / Ethnicity",
    "Sex",
    "Employment",
    "Tax Status",
    "Household",
    "Migration",
    "Family",
    "Country of Birth",
    "Citizenship",
    "Other"
]

row_labels = [
    "CFS (10)", "CFS (30)", "CFS (50)", "CFS (100)",
    "KBest (10)", "KBest (30)", "KBest (50)", "KBest (100)",
    "RFE (10)", "RFE (30)", "RFE (50)", "RFE (100)"
]

# --------------------------------------------------
# Feature composition data
# --------------------------------------------------

data = np.array([
    # CFS
    [2, 1, 0, 2, 3, 0, 0, 2, 0, 0, 0, 0, 1, 0, 0, 0, 0, 0],
    [2, 3, 3, 6, 5, 0, 1, 2, 1, 1, 1, 2, 2, 0, 0, 0, 0, 0],
    [2, 3, 8, 9, 9, 1, 2, 2, 1, 2, 1, 2, 2, 0, 2, 1, 0, 0],
    [3, 4, 12, 19, 13, 3, 5, 2, 3, 2, 2, 2, 5, 3, 3, 11, 0, 0],

    # KBest
    [9, 6, 19, 17, 11, 3, 12, 7, 2, 2, 4, 2, 6, 3, 0, 3, 2, 2],
    [4, 1, 1, 4, 2, 1, 1, 3, 0, 0, 0, 2, 2, 0, 0, 0, 0, 0],
    [7, 4, 5, 7, 8, 2, 4, 2, 2, 2, 3, 2, 4, 0, 2, 1, 0, 0],
    [7, 6, 8, 14, 11, 2, 10, 7, 3, 2, 4, 2, 5, 0, 3, 5, 2, 2],

    # RFE
    [6, 0, 0, 0, 0, 0, 0, 2, 0, 1, 0, 0, 1, 0, 0, 0, 0, 0],
    [5, 2, 1, 4, 4, 2, 1, 3, 0, 2, 1, 2, 3, 0, 1, 0, 0, 0],
    [8, 4, 0, 4, 7, 2, 5, 5, 1, 2, 2, 1, 2, 2, 1, 2, 1, 1],
    [9, 5, 10, 14, 8, 3, 10, 7, 3, 2, 3, 2, 6, 2, 3, 3, 2, 2]
])

# --------------------------------------------------
# Custom colormap
# --------------------------------------------------

cmap = LinearSegmentedColormap.from_list(
    "custom_heatmap",
    [
        "#FFFFFF",  # Zero: white
        "#FFF7BC",  # Very low: pale yellow
        "#FEC44F",  # Low: yellow-orange
        "#FE9929",  # Medium: orange
        "#EC7014",  # High: dark orange
        "#CC4C02",  # Very high: red-orange
        "#8C2D04"   # Maximum: dark red
    ],
    N=256
)

# Increase contrast among values
norm = PowerNorm(
    gamma=0.65,
    vmin=0,
    vmax=data.max()
)

# --------------------------------------------------
# Plot
# --------------------------------------------------

fig, ax = plt.subplots(figsize=(17, 8))

im = ax.imshow(
    data,
    cmap=cmap,
    norm=norm,
    aspect="auto",
    interpolation="nearest"
)

# X-axis
ax.set_xticks(np.arange(len(feature_groups)))
ax.set_xticklabels(
    feature_groups,
    rotation=40,
    ha="right",
    fontsize=10
)

# Y-axis
ax.set_yticks(np.arange(len(row_labels)))
ax.set_yticklabels(
    row_labels,
    fontsize=10
)

# Title and labels
ax.set_title(
    "Feature Composition Across Selection Strategies",
    fontsize=15,
    fontweight="bold",
    pad=18
)

ax.set_xlabel(
    "Feature Groups",
    fontsize=11,
    labelpad=12
)

ax.set_ylabel(
    "Selection Strategy and Subset Size",
    fontsize=11
)

# --------------------------------------------------
# Cell annotations
# --------------------------------------------------

for i in range(data.shape[0]):
    for j in range(data.shape[1]):

        value = data[i, j]

        ax.text(
            j,
            i,
            str(value),
            ha="center",
            va="center",
            fontsize=8,
            fontweight="medium",
            color="white" if value >= 10 else "black"
        )

# --------------------------------------------------
# Separate feature selection methods
# --------------------------------------------------

for y in [3.5, 7.5]:
    ax.axhline(
        y,
        color="#000000",
        linewidth=3,
        linestyle="-",
    )

# --------------------------------------------------
# Grid
# --------------------------------------------------

ax.set_xticks(
    np.arange(-0.5, len(feature_groups), 1),
    minor=True
)

ax.set_yticks(
    np.arange(-0.5, len(row_labels), 1),
    minor=True
)

ax.grid(
    which="minor",
    color="white",
    linewidth=1
)

ax.tick_params(
    which="minor",
    bottom=False,
    left=False
)

# --------------------------------------------------
# Colorbar
# --------------------------------------------------

cbar = fig.colorbar(
    im,
    ax=ax,
    fraction=0.025,
    pad=0.02
)

cbar.set_label(
    "Number of Selected Features",
    fontsize=10
)

# --------------------------------------------------
# Appearance
# --------------------------------------------------

ax.spines["top"].set_visible(False)
ax.spines["right"].set_visible(False)

plt.tight_layout()

# --------------------------------------------------
# Save figure
# --------------------------------------------------

plt.savefig(
    "feature_composition_heatmap.png",
    dpi=300,
    bbox_inches="tight",
    transparent=True
)

plt.show()