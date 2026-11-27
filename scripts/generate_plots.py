"""Script to generate evaluation and architectural plots for RepoLens.

Outputs publication-quality PNG and SVG figures to assets/.
"""

from __future__ import annotations

import json
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import seaborn as sns

REPO_ROOT = Path(__file__).resolve().parent.parent
ASSETS_DIR = REPO_ROOT / "assets"
REPORT_PATH = REPO_ROOT / "dataset" / "evaluation" / "benchmark_report.json"

# Set style theme
plt.style.use("seaborn-v0_8-whitegrid" if "seaborn-v0_8-whitegrid" in plt.style.available else "default")
plt.rcParams["font.family"] = "sans-serif"
plt.rcParams["font.sans-serif"] = ["DejaVu Sans", "Arial", "Helvetica"]
plt.rcParams["axes.edgecolor"] = "#cccccc"
plt.rcParams["axes.linewidth"] = 0.8


def load_benchmark_data() -> dict:
    """Load benchmark report JSON."""
    with open(REPORT_PATH, "r", encoding="utf-8") as f:
        return json.load(f)


def plot_baseline_comparison(data: dict) -> None:
    """Plot 1: Grouped bar chart of core retrieval metrics across B0-B5."""
    configs = data["configurations"]
    names = ["B0: Naive", "B1: AST", "B2: Hybrid", "B3: Graph", "B4: Agentic", "B5: RepoLens"]

    recall = [c["mean_recall_at_5"] for c in configs]
    precision = [c["mean_precision_at_5"] for c in configs]
    mrr = [c["mean_mrr"] for c in configs]
    ndcg = [c["mean_ndcg_at_5"] for c in configs]

    x = np.arange(len(names))
    width = 0.2

    fig, ax = plt.subplots(figsize=(10, 5.5), dpi=300)

    rects1 = ax.bar(x - 1.5 * width, recall, width, label="Recall@5", color="#2b5c8f")
    rects2 = ax.bar(x - 0.5 * width, precision, width, label="Precision@5", color="#3b82f6")
    rects3 = ax.bar(x + 0.5 * width, mrr, width, label="MRR", color="#10b981")
    rects4 = ax.bar(x + 1.5 * width, ndcg, width, label="nDCG@5", color="#f59e0b")

    ax.set_ylabel("Score (0.0 – 1.0)", fontsize=11, fontweight="bold")
    ax.set_title("RepoLens: Retrieval Quality Across Incremental Baselines (B0 to B5)", fontsize=13, fontweight="bold", pad=15)
    ax.set_xticks(x)
    ax.set_xticklabels(names, fontsize=10, fontweight="bold")
    ax.set_ylim(0, 1.15)
    ax.legend(loc="upper left", frameon=True, framealpha=0.9, fontsize=10)
    ax.grid(axis="y", linestyle="--", alpha=0.7)

    # Add value labels on top of Recall@5 bars
    for rect in rects1:
        height = rect.get_height()
        ax.annotate(
            f"{height:.2f}",
            xy=(rect.get_x() + rect.get_width() / 2, height),
            xytext=(0, 3),
            textcoords="offset points",
            ha="center",
            va="bottom",
            fontsize=8,
            fontweight="bold",
            color="#2b5c8f",
        )

    plt.tight_layout()
    fig.savefig(ASSETS_DIR / "benchmark_baseline_comparison.png", dpi=300)
    fig.savefig(ASSETS_DIR / "benchmark_baseline_comparison.svg")
    plt.close(fig)
    print("Saved benchmark_baseline_comparison.png / .svg")


def plot_citation_precision_and_hallucination(data: dict) -> None:
    """Plot 2: Dual panel showing Citation Precision vs Unsupported Hallucination Rate."""
    configs = data["configurations"]
    names = ["B0: Naive", "B1: AST", "B2: Hybrid", "B3: Graph", "B4: Agentic", "B5: RepoLens"]

    prec = [c["mean_citation_precision"] * 100 for c in configs]
    hallucination = [100.0 - p for p in prec]

    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(11, 4.5), dpi=300)

    # Panel 1: Citation Precision
    colors1 = ["#ef4444" if p < 99 else "#10b981" for p in prec]
    bars1 = ax1.bar(names, prec, color=colors1, width=0.55)
    ax1.set_ylabel("Citation Precision (%)", fontsize=11, fontweight="bold")
    ax1.set_title("Source Citation Precision", fontsize=12, fontweight="bold")
    ax1.set_ylim(0, 115)
    ax1.grid(axis="y", linestyle="--", alpha=0.7)

    for bar in bars1:
        h = bar.get_height()
        ax1.annotate(f"{h:.1f}%", xy=(bar.get_x() + bar.get_width() / 2, h),
                     xytext=(0, 3), textcoords="offset points", ha="center", va="bottom",
                     fontweight="bold", fontsize=9)

    # Panel 2: Unsupported Citation Rate (Hallucinations)
    colors2 = ["#ef4444" if h > 1 else "#10b981" for h in hallucination]
    bars2 = ax2.bar(names, hallucination, color=colors2, width=0.55)
    ax2.set_ylabel("Unsupported Citation Rate (%)", fontsize=11, fontweight="bold")
    ax2.set_title("Hallucinated / Invalid Citations Rate", fontsize=12, fontweight="bold")
    ax2.set_ylim(0, 70)
    ax2.grid(axis="y", linestyle="--", alpha=0.7)

    for bar in bars2:
        h = bar.get_height()
        ax2.annotate(f"{h:.1f}%", xy=(bar.get_x() + bar.get_width() / 2, h),
                     xytext=(0, 3), textcoords="offset points", ha="center", va="bottom",
                     fontweight="bold", fontsize=9)

    fig.suptitle("Impact of AST Parsing on Eliminating Citation Hallucinations", fontsize=13, fontweight="bold", y=1.02)
    plt.tight_layout()
    fig.savefig(ASSETS_DIR / "citation_precision_hallucination.png", dpi=300)
    fig.savefig(ASSETS_DIR / "citation_precision_hallucination.svg")
    plt.close(fig)
    print("Saved citation_precision_hallucination.png / .svg")


def plot_latency_vs_accuracy_pareto(data: dict) -> None:
    """Plot 3: Scatter plot of Latency (ms) vs Recall@5 with Cost annotations."""
    configs = data["configurations"]
    names = ["B0: Naive RAG", "B1: AST-aware RAG", "B2: Hybrid RAG", "B3: Graph RAG", "B4: Agentic RAG", "B5: Full RepoLens"]

    latencies = [c["mean_latency_ms"] for c in configs]
    recalls = [c["mean_recall_at_5"] for c in configs]
    costs = [0.0038, 0.0029, 0.0118, 0.0136, 0.0193, 0.0223]  # Estimated cost in USD

    fig, ax = plt.subplots(figsize=(8.5, 5.2), dpi=300)

    colors = ["#94a3b8", "#3b82f6", "#06b6d4", "#f59e0b", "#ec4899", "#10b981"]

    for i in range(len(names)):
        ax.scatter(latencies[i], recalls[i], s=costs[i] * 12000 + 100, color=colors[i], edgecolors="#1e293b", linewidth=1.5, zorder=4)
        offset_y = 0.03 if i % 2 == 0 else -0.05
        offset_x = 10 if i != 5 else -70
        ax.annotate(
            f"{names[i]}\n({latencies[i]:.0f}ms, ${costs[i]:.4f})",
            xy=(latencies[i], recalls[i]),
            xytext=(latencies[i] + offset_x, recalls[i] + offset_y),
            fontsize=8.5,
            fontweight="bold",
            arrowprops=dict(arrowstyle="->", color="#64748b", lw=0.8),
            zorder=5,
        )

    # Plot Pareto trajectory curve
    ax.plot(latencies, recalls, linestyle=":", color="#94a3b8", zorder=3)

    ax.set_xlabel("Mean Latency per Query (milliseconds)", fontsize=11, fontweight="bold")
    ax.set_ylabel("Gold Evidence Recall@5", fontsize=11, fontweight="bold")
    ax.set_title("Efficiency vs. Retrieval Accuracy Trade-off (Pareto Frontier)", fontsize=13, fontweight="bold", pad=15)
    ax.set_ylim(0.25, 1.12)
    ax.set_xlim(20, 850)
    ax.grid(True, linestyle="--", alpha=0.6)

    # Annotate bubble size
    ax.text(0.03, 0.05, "Bubble size proportional to total token cost ($)", transform=ax.transAxes, fontsize=8.5, style="italic", bbox=dict(boxstyle="round", facecolor="#f8fafc", edgecolor="#cbd5e1"))

    plt.tight_layout()
    fig.savefig(ASSETS_DIR / "latency_vs_accuracy_pareto.png", dpi=300)
    fig.savefig(ASSETS_DIR / "latency_vs_accuracy_pareto.svg")
    plt.close(fig)
    print("Saved latency_vs_accuracy_pareto.png / .svg")


def plot_component_ablation_contributions() -> None:
    """Plot 4: Waterfall chart showing marginal recall gains from each component."""
    labels = [
        "Naive RAG (Baseline)",
        "+ AST Structural Partitioning",
        "+ Hybrid BM25 & Semantic Fusion",
        "+ Code Graph Context Expansion",
        "+ Multi-Turn LangGraph Agent",
        "Full RepoLens (Total)",
    ]

    base = 0.3742
    g_ast = 0.4608 - 0.3742
    g_hybrid = 0.6817 - 0.4608
    g_graph = 0.9667 - 0.6817
    g_agent = 1.0000 - 0.9667
    total = 1.0000

    values = [base, g_ast, g_hybrid, g_graph, g_agent, total]
    colors = ["#64748b", "#3b82f6", "#06b6d4", "#f59e0b", "#ec4899", "#10b981"]

    fig, ax = plt.subplots(figsize=(9, 5), dpi=300)
    y_pos = np.arange(len(labels))

    bars = ax.barh(y_pos, values, color=colors, height=0.55)
    ax.set_yticks(y_pos)
    ax.set_yticklabels(labels, fontsize=10, fontweight="bold")
    ax.invert_yaxis()  # Labels read top-to-bottom
    ax.set_xlabel("Gold Evidence Recall@5 Contribution", fontsize=11, fontweight="bold")
    ax.set_title("Marginal Recall Gains Across Architectural Components", fontsize=13, fontweight="bold", pad=15)
    ax.set_xlim(0, 1.15)
    ax.grid(axis="x", linestyle="--", alpha=0.7)

    for i, bar in enumerate(bars):
        w = bar.get_width()
        prefix = "+" if 0 < i < 5 else ""
        ax.annotate(
            f"{prefix}{w*100:.2f}%",
            xy=(w, bar.get_y() + bar.get_height() / 2),
            xytext=(5, 0),
            textcoords="offset points",
            va="center",
            fontweight="bold",
            fontsize=9.5,
        )

    plt.tight_layout()
    fig.savefig(ASSETS_DIR / "component_ablation_contributions.png", dpi=300)
    fig.savefig(ASSETS_DIR / "component_ablation_contributions.svg")
    plt.close(fig)
    print("Saved component_ablation_contributions.png / .svg")


def plot_per_repository_performance(data: dict) -> None:
    """Plot 5: Grouped comparison across the 5 repositories for B0 vs B5."""
    b0 = data["configurations"][0]["per_repo_metrics"]
    b5 = data["configurations"][5]["per_repo_metrics"]

    repos = ["requests", "flask", "click", "fastapi", "pydantic"]
    repo_labels = ["Requests\n(HTTP)", "Flask\n(WSGI)", "Click\n(CLI)", "FastAPI\n(Async/DI)", "Pydantic\n(Validation)"]

    b0_recalls = [b0[r]["recall_at_5"] * 100 for r in repos]
    b5_recalls = [b5[r]["recall_at_5"] * 100 for r in repos]

    x = np.arange(len(repos))
    width = 0.35

    fig, ax = plt.subplots(figsize=(9, 4.8), dpi=300)

    rects1 = ax.bar(x - width / 2, b0_recalls, width, label="B0: Naive RAG", color="#94a3b8")
    rects2 = ax.bar(x + width / 2, b5_recalls, width, label="B5: Full RepoLens", color="#10b981")

    ax.set_ylabel("Gold Evidence Recall@5 (%)", fontsize=11, fontweight="bold")
    ax.set_title("Cross-Repository Generalizability (B0 Naive vs B5 RepoLens)", fontsize=13, fontweight="bold", pad=15)
    ax.set_xticks(x)
    ax.set_xticklabels(repo_labels, fontsize=10, fontweight="bold")
    ax.set_ylim(0, 120)
    ax.legend(loc="upper left", frameon=True, fontsize=10)
    ax.grid(axis="y", linestyle="--", alpha=0.7)

    for rect in rects1:
        h = rect.get_height()
        ax.annotate(f"{h:.1f}%", xy=(rect.get_x() + rect.get_width() / 2, h), xytext=(0, 3), textcoords="offset points", ha="center", va="bottom", fontsize=8.5, fontweight="bold", color="#475569")

    for rect in rects2:
        h = rect.get_height()
        ax.annotate(f"{h:.1f}%", xy=(rect.get_x() + rect.get_width() / 2, h), xytext=(0, 3), textcoords="offset points", ha="center", va="bottom", fontsize=8.5, fontweight="bold", color="#059669")

    plt.tight_layout()
    fig.savefig(ASSETS_DIR / "per_repository_performance.png", dpi=300)
    fig.savefig(ASSETS_DIR / "per_repository_performance.svg")
    plt.close(fig)
    print("Saved per_repository_performance.png / .svg")


def main() -> None:
    data = load_benchmark_data()
    print("Generating RepoLens plots in assets/...")
    plot_baseline_comparison(data)
    plot_citation_precision_and_hallucination(data)
    plot_latency_vs_accuracy_pareto(data)
    plot_component_ablation_contributions()
    plot_per_repository_performance(data)
    print("All plots generated successfully!")


if __name__ == "__main__":
    main()
