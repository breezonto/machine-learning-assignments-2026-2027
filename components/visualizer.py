import pandas as pd
import matplotlib.pyplot as plt
import numpy as np


def plot_classifier_performance(csv_path):
    # Read classification results
    df = pd.read_csv(csv_path)

    # Classifier names
    models = df["model"].tolist()

    # Metrics to compare
    metrics = {
        "Accuracy": "accuracy",
        "Macro Precision": "precision_macro",
        "Macro Recall": "recall_macro",
        "Macro F1": "f1_macro",
        "Class 0 F1": "0_f1-score",
        "Class 1 F1": "1_f1-score"
    }

    # Plot configuration
    x = np.arange(len(models))
    n_metrics = len(metrics)
    width = 0.12

    fig, ax = plt.subplots(figsize=(13, 7))

    # Draw grouped bars
    for i, (label, column) in enumerate(metrics.items()):
        offset = (i - (n_metrics - 1) / 2) * width

        ax.bar(
            x + offset,
            df[column],
            width,
            label=label
        )

    # Formatting
    ax.set_title("Classifier Performance Comparison", fontsize=16)
    ax.set_xlabel("Classification Models", fontsize=12)
    ax.set_ylabel("Performance Score", fontsize=12)

    ax.set_xticks(x)
    ax.set_xticklabels(models, rotation=15, ha="right")

    ax.set_ylim(0, 1.05)
    ax.grid(axis="y", linestyle="--", alpha=0.4)
    ax.set_axisbelow(True)

    ax.legend(
        title="Evaluation Metrics",
        loc="upper center",
        bbox_to_anchor=(0.5, -0.18),
        ncol=3,
        frameon=False
    )

    plt.tight_layout()
    plt.show()


# Usage
plot_classifier_performance("results/results.csv")