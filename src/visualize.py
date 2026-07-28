"""
visualize.py
------------
Publication-quality visualizations for the Bayesian Network model.


Generates:
- Network DAG rendering
- Confusion matrix heatmap
- Calibration (reliability) diagram
- Predicted probability distribution bar charts
- Inference method comparison plots
"""


import os
import logging


import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
import seaborn as sns


from src.network_structure import (
    NODES, TOPOLOGICAL_ORDER, DISPOSITION_LABELS, BINARY_LABELS,
)


log = logging.getLogger(__name__)


# Consistent style
plt.rcParams.update({
    "figure.dpi": 150,
    "savefig.dpi": 150,
    "font.size": 10,
    "axes.titlesize": 12,
    "figure.figsize": (10, 6),
})


# figures/ lives at the repository root, not inside the package.
OUTPUT_DIR = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "figures")



def _ensure_output_dir():
    os.makedirs(OUTPUT_DIR, exist_ok=True)



# ---------------------------------------------------------------------------
# 1. Network DAG
# ---------------------------------------------------------------------------


def plot_network_dag(save: bool = True):
    """
    Render the Bayesian Network as a layered DAG using matplotlib.


    Nodes are color-coded by layer:
    - Green: Root nodes (court info + case properties)
    - Blue: Intermediate nodes
    - Red: Final outcome
    """
    _ensure_output_dir()


    layer_positions = {
        "chief_justice": (0.5, 4),
        "issue_area": (0, 3),
        "law_type": (1, 3),
        "case_supplement": (2, 3),
        "lower_court_disposition": (3, 3),
        "decision_type": (0.5, 2),
        "precedent_alteration": (1.5, 2),
        "split_vote": (2.5, 2),
        "unconstitutional": (3.5, 2),
        "final_disposition": (2, 1),
    }


    layer_colors = {
        "chief_justice": "#66bb6a",
        "issue_area": "#66bb6a",
        "law_type": "#66bb6a",
        "case_supplement": "#66bb6a",
        "lower_court_disposition": "#66bb6a",
        "decision_type": "#42a5f5",
        "precedent_alteration": "#42a5f5",
        "split_vote": "#42a5f5",
        "unconstitutional": "#42a5f5",
        "final_disposition": "#ef5350",
    }


    fig, ax = plt.subplots(1, 1, figsize=(12, 8))


    # Draw edges first
    for node_name in TOPOLOGICAL_ORDER:
        node = NODES[node_name]
        x1, y1 = layer_positions[node_name]
        for parent_name in node.parents:
            x0, y0 = layer_positions[parent_name]
            ax.annotate(
                "",
                xy=(x1, y1 + 0.15),
                xytext=(x0, y0 - 0.15),
                arrowprops=dict(arrowstyle="->", color="#666", lw=1.5, connectionstyle="arc3,rad=0.1"),
            )


    # Draw nodes
    for node_name in TOPOLOGICAL_ORDER:
        x, y = layer_positions[node_name]
        color = layer_colors[node_name]
        label = node_name.replace("_", "\n")
        # FancyBboxPatch, not Rectangle: only the former accepts `boxstyle`,
        # and passing it to a plain Rectangle raises.
        ax.add_patch(mpatches.FancyBboxPatch(
            (x - 0.4, y - 0.12), 0.8, 0.24,
            boxstyle="round,pad=0.02",
            facecolor=color, edgecolor="#333",
            linewidth=1.5, zorder=3, alpha=0.9,
        ))
        ax.text(x, y, label, ha="center", va="center", fontsize=8,
                fontweight="bold", zorder=4)


    # Legend
    legend_elements = [
        mpatches.Patch(facecolor="#66bb6a", label="Root Nodes"),
        mpatches.Patch(facecolor="#42a5f5", label="Intermediate Nodes"),
        mpatches.Patch(facecolor="#ef5350", label="Final Outcome"),
    ]
    ax.legend(handles=legend_elements, loc="upper right", fontsize=9)


    ax.set_xlim(-0.8, 4.3)
    ax.set_ylim(0.5, 4.5)
    ax.set_aspect("equal")
    ax.axis("off")
    ax.set_title("Bayesian Network Structure — Supreme Court Decision Prediction", fontsize=14)


    plt.tight_layout()
    if save:
        path = os.path.join(OUTPUT_DIR, "network_dag.png")
        fig.savefig(path, bbox_inches="tight")
        print(f"  Saved: {path}")
    plt.close(fig)
    return fig



# ---------------------------------------------------------------------------
# 2. Confusion Matrix
# ---------------------------------------------------------------------------


def plot_confusion_matrix(
    result: dict,
    filename: str = "confusion_matrix.png",
    save: bool = True,
):
    """
    Heatmap confusion matrix for top-1 predictions.

    Takes an `evaluate_model` result rather than a sampler, so it reuses the
    posterior matrix already computed instead of running inference again.
    """
    _ensure_output_dir()

    y_true = result["_y_true"]
    P = result["_P"]
    classes = result["classes"]
    labels_map = BINARY_LABELS if result.get("binary") else DISPOSITION_LABELS

    class_array = np.array(classes, dtype=object)
    predicted = class_array[np.argmax(P, axis=1)]

    index = {c: i for i, c in enumerate(classes)}
    matrix = np.zeros((len(classes), len(classes)), dtype=int)
    for true_val, pred_val in zip(y_true, predicted):
        if true_val in index and pred_val in index:
            matrix[index[true_val], index[pred_val]] += 1

    def _label(c):
        try:
            return labels_map.get(int(c), str(c))[:20]
        except (TypeError, ValueError):
            return str(c)[:20]

    labels = [_label(c) for c in classes]

    fig, ax = plt.subplots(figsize=(10, 8))
    sns.heatmap(matrix, annot=True, fmt="d", cmap="Blues",
                xticklabels=labels, yticklabels=labels, ax=ax)
    ax.set_xlabel("Predicted")
    ax.set_ylabel("Actual")
    ax.set_title(f"Confusion Matrix — {result.get('name', 'model')} (Top-1)")
    plt.xticks(rotation=45, ha="right")
    plt.yticks(rotation=0)
    plt.tight_layout()

    if save:
        path = os.path.join(OUTPUT_DIR, filename)
        fig.savefig(path, bbox_inches="tight")
        print(f"  Saved: {path}")
    plt.close(fig)
    return fig


# ---------------------------------------------------------------------------
# 3. Calibration (Reliability) Diagram
# ---------------------------------------------------------------------------


def plot_calibration_diagram(cal: dict, save: bool = True):
    """
    Plot a reliability diagram from calibration analysis results.


    Perfect calibration = diagonal line. Bars above the diagonal indicate
    underconfidence; below indicates overconfidence.
    """
    _ensure_output_dir()


    edges = cal["bin_edges"]
    accs = cal["bin_accuracies"]
    confs = cal["bin_confidences"]
    counts = cal["bin_counts"]


    fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(8, 8), gridspec_kw={"height_ratios": [3, 1]})


    # Reliability diagram
    bin_centers = [(edges[i] + edges[i + 1]) / 2 for i in range(len(accs))]
    ax1.bar(bin_centers, accs, width=0.08, alpha=0.7, color="#42a5f5", label="Accuracy", edgecolor="#333")
    ax1.plot([0, 1], [0, 1], "k--", lw=1.5, label="Perfect calibration")
    ax1.set_xlabel("Mean Predicted Confidence")
    ax1.set_ylabel("Fraction of Positives")
    ax1.set_title(f"Calibration Diagram (ECE = {cal['ece']:.4f})")
    ax1.legend(loc="upper left")
    ax1.set_xlim(0, 1)
    ax1.set_ylim(0, 1)


    # Histogram of predictions
    ax2.bar(bin_centers, counts, width=0.08, alpha=0.7, color="#ef5350", edgecolor="#333")
    ax2.set_xlabel("Predicted Confidence")
    ax2.set_ylabel("Count")
    ax2.set_title("Prediction Distribution")


    plt.tight_layout()
    if save:
        path = os.path.join(OUTPUT_DIR, "calibration_diagram.png")
        fig.savefig(path, bbox_inches="tight")
        print(f"  Saved: {path}")
    plt.close(fig)
    return fig



# ---------------------------------------------------------------------------
# 4. Probability Distribution for a Single Case
# ---------------------------------------------------------------------------


def plot_prediction_distribution(
    distribution: dict,
    true_val=None,
    title: str = "Predicted Disposition Distribution",
    save: bool = True,
    filename: str = "prediction_distribution.png",
):
    """Bar chart of predicted probabilities across all disposition classes."""
    _ensure_output_dir()


    values = sorted(distribution.keys())
    probs = [distribution[v] for v in values]
    labels = [DISPOSITION_LABELS.get(int(v), str(v))[:25] for v in values]


    colors = ["#ef5350" if v == true_val else "#42a5f5" for v in values]


    fig, ax = plt.subplots(figsize=(12, 6))
    bars = ax.barh(range(len(values)), probs, color=colors, edgecolor="#333")
    ax.set_yticks(range(len(values)))
    ax.set_yticklabels(labels, fontsize=9)
    ax.set_xlabel("Probability")
    ax.set_title(title)
    ax.invert_yaxis()


    for bar, prob in zip(bars, probs):
        ax.text(bar.get_width() + 0.01, bar.get_y() + bar.get_height() / 2,
                f"{prob:.3f}", va="center", fontsize=8)


    if true_val is not None:
        ax.legend([mpatches.Patch(color="#ef5350"), mpatches.Patch(color="#42a5f5")],
                  ["True Class", "Predicted"], loc="lower right")


    plt.tight_layout()
    if save:
        path = os.path.join(OUTPUT_DIR, filename)
        fig.savefig(path, bbox_inches="tight")
        print(f"  Saved: {path}")
    plt.close(fig)
    return fig



# ---------------------------------------------------------------------------
# 5. Inference Method Comparison
# ---------------------------------------------------------------------------


def plot_method_comparison(results: dict, save: bool = True):
    """Bar chart comparing accuracy and speed across inference methods."""
    _ensure_output_dir()


    methods = list(results.keys())
    accuracies = [results[m]["accuracy"] * 100 for m in methods]
    times = [results[m]["time_seconds"] for m in methods]


    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 5))


    # Accuracy
    bars1 = ax1.bar(methods, accuracies, color=["#66bb6a", "#42a5f5", "#ff9800"],
                    edgecolor="#333")
    ax1.set_ylabel("Top-k Accuracy (%)")
    ax1.set_title("Accuracy Comparison")
    for bar, acc in zip(bars1, accuracies):
        ax1.text(bar.get_x() + bar.get_width() / 2, bar.get_height() + 0.5,
                 f"{acc:.1f}%", ha="center", fontsize=10)


    # Time
    bars2 = ax2.bar(methods, times, color=["#66bb6a", "#42a5f5", "#ff9800"],
                    edgecolor="#333")
    ax2.set_ylabel("Time (seconds)")
    ax2.set_title("Speed Comparison")
    for bar, t in zip(bars2, times):
        ax2.text(bar.get_x() + bar.get_width() / 2, bar.get_height() + 0.1,
                 f"{t:.1f}s", ha="center", fontsize=10)


    plt.suptitle("Inference Method Comparison", fontsize=14, y=1.02)
    plt.tight_layout()


    if save:
        path = os.path.join(OUTPUT_DIR, "method_comparison.png")
        fig.savefig(path, bbox_inches="tight")
        print(f"  Saved: {path}")
    plt.close(fig)
    return fig



# ---------------------------------------------------------------------------
# 6. Dependency Heatmap
# ---------------------------------------------------------------------------


def plot_dependency_heatmap(dep_matrix: pd.DataFrame, method: str = "deviation", save: bool = True):
    """Heatmap of pairwise variable dependencies."""
    _ensure_output_dir()


    fig, ax = plt.subplots(figsize=(10, 8))
    label = "D(A,B)" if method == "deviation" else "I(A;B)"
    sns.heatmap(dep_matrix, annot=True, fmt=".4f", cmap="YlOrRd",
                ax=ax, linewidths=0.5)
    ax.set_title(f"Pairwise Variable Dependencies — {label}")
    plt.xticks(rotation=45, ha="right")
    plt.yticks(rotation=0)
    plt.tight_layout()


    if save:
        path = os.path.join(OUTPUT_DIR, f"dependency_heatmap_{method}.png")
        fig.savefig(path, bbox_inches="tight")
        print(f"  Saved: {path}")
    plt.close(fig)
    return fig