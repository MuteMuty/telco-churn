"""Charts for the exploratory analysis and model evaluation."""

import math

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from matplotlib.axes import Axes
from matplotlib.figure import Figure
from matplotlib.ticker import PercentFormatter

from telco_churn.data import TARGET

STAYED_COLOR = "#2a78d6"
CHURNED_COLOR = "#eb6834"
TEXT_COLOR = "#0b0b0b"
MUTED_COLOR = "#52514e"
GRID_COLOR = "#e4e3df"


def set_style() -> None:
    """Quiet axes and grid so the data stands out."""
    plt.rcParams.update(
        {
            "figure.dpi": 110,
            "font.size": 10,
            "text.color": TEXT_COLOR,
            "axes.labelcolor": MUTED_COLOR,
            "axes.edgecolor": GRID_COLOR,
            "axes.spines.top": False,
            "axes.spines.right": False,
            "axes.grid": True,
            "axes.axisbelow": True,
            "axes.titlelocation": "left",
            "axes.titlesize": 11,
            "axes.titleweight": "bold",
            "grid.color": GRID_COLOR,
            "grid.linewidth": 0.8,
            "xtick.color": MUTED_COLOR,
            "ytick.color": MUTED_COLOR,
            "legend.frameon": False,
        }
    )


def churn_rate_by(df: pd.DataFrame, column: str) -> pd.DataFrame:
    """Number of customers and churn rate for each value of `column`, highest rate first."""
    return (
        df.groupby(column, observed=True)[TARGET]
        .agg(customers="size", churn_rate="mean")
        .sort_values("churn_rate", ascending=False)
    )


def plot_target_distribution(y: pd.Series, ax: Axes | None = None) -> Axes:
    """Bar chart of customers who stayed vs churned."""
    ax = ax or plt.subplots(figsize=(5, 3.5))[1]
    counts = y.value_counts().reindex([0, 1])

    bars = ax.bar(
        ["Stayed", "Churned"], counts, color=[STAYED_COLOR, CHURNED_COLOR], width=0.5
    )
    ax.bar_label(bars, labels=[f"{n:,} ({n / len(y):.1%})" for n in counts], padding=3)

    ax.set_title("Churn distribution")
    ax.set_ylabel("Customers")
    ax.set_ylim(0, counts.max() * 1.15)
    ax.grid(axis="x", visible=False)
    return ax


def plot_churn_rate(
    df: pd.DataFrame, column: str, ax: Axes | None = None, xmax: float | None = None
) -> Axes:
    """Horizontal bars of churn rate per category, with the overall rate as a dashed line."""
    # barh draws from the bottom up, so reverse to put the highest rate on top
    rates = churn_rate_by(df, column).iloc[::-1]
    overall = df[TARGET].mean()
    ax = ax or plt.subplots(figsize=(6, 0.5 * len(rates) + 1))[1]

    labels = [f"{value}  (n={n:,})" for value, n in zip(rates.index, rates["customers"])]
    bars = ax.barh(labels, rates["churn_rate"], color=CHURNED_COLOR, height=0.6)
    ax.bar_label(bars, labels=[f"{r:.0%}" for r in rates["churn_rate"]], padding=3)
    ax.axvline(overall, color=MUTED_COLOR, linestyle="--", linewidth=1)

    ax.set_title(column)
    ax.set_xlabel(f"Churn rate (dashed line = overall {overall:.1%})")
    ax.set_xlim(0, xmax or rates["churn_rate"].max() * 1.2)
    ax.xaxis.set_major_formatter(PercentFormatter(1.0))
    ax.grid(axis="y", visible=False)
    return ax


def plot_churn_rate_grid(
    df: pd.DataFrame, columns: list[str], ncols: int = 3
) -> Figure:
    """Small multiples of `plot_churn_rate`, one per column, on a shared x scale."""
    nrows = math.ceil(len(columns) / ncols)
    fig, axes = plt.subplots(nrows, ncols, figsize=(5.5 * ncols, 2.3 * nrows))
    # same x scale everywhere, so bar lengths are comparable across panels
    xmax = max(churn_rate_by(df, c)["churn_rate"].max() for c in columns) * 1.2

    for ax, column in zip(axes.flat, columns):
        plot_churn_rate(df, column, ax=ax, xmax=xmax)
        ax.set_xlabel("")
    for ax in axes.flat[len(columns) :]:
        ax.set_visible(False)

    overall = df[TARGET].mean()
    fig.suptitle(
        f"Churn rate by category (dashed line = overall {overall:.1%})",
        x=0.01,
        ha="left",
        fontweight="bold",
    )
    fig.tight_layout()
    return fig


def plot_numeric_by_churn(
    df: pd.DataFrame, column: str, ax: Axes | None = None, bins: int = 30
) -> Axes:
    """Distribution of a numeric column for stayed vs churned customers.

    Each group is normalised to its own total (density), so the smaller churned
    group is comparable to the larger stayed group.
    """
    ax = ax or plt.subplots(figsize=(5, 3.5))[1]
    edges = np.histogram_bin_edges(df[column], bins=bins)

    for value, label, color in [(0, "Stayed", STAYED_COLOR), (1, "Churned", CHURNED_COLOR)]:
        ax.hist(
            df.loc[df[TARGET] == value, column],
            bins=edges,
            density=True,
            histtype="step",
            linewidth=2,
            color=color,
            label=label,
        )

    ax.set_title(column)
    ax.set_ylabel("Density")
    ax.set_yticks([])
    ax.legend()
    return ax
