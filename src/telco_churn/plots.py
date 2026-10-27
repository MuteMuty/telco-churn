"""Charts for the exploratory analysis and model evaluation."""

import math

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from matplotlib.axes import Axes
from matplotlib.figure import Figure
from matplotlib.ticker import FuncFormatter, PercentFormatter
from sklearn.calibration import CalibrationDisplay
from sklearn.metrics import (
    ConfusionMatrixDisplay,
    PrecisionRecallDisplay,
    RocCurveDisplay,
)

from telco_churn.data import TARGET

STAYED_COLOR = "#2a78d6"
CHURNED_COLOR = "#eb6834"
NEUTRAL_COLOR = "#8a8983"
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


def churn_rate_by(
    df: pd.DataFrame, column: str, sort_by_rate: bool = True
) -> pd.DataFrame:
    """Number of customers and churn rate for each value of `column`.

    Sorted by churn rate (highest first), or by the values of `column` if
    `sort_by_rate` is False (useful for ordered values such as counts).
    """
    rates = df.groupby(column, observed=True)[TARGET].agg(
        customers="size", churn_rate="mean"
    )
    return rates.sort_values("churn_rate", ascending=False) if sort_by_rate else rates


def churn_rate_gaps(df: pd.DataFrame, columns: list[str]) -> pd.Series:
    """Gap between the highest and lowest churn rate within each column, largest first.

    A rough first look at how strongly a feature separates churners; it ignores
    overlap between features, so it is not a measure of feature importance.
    """
    gaps = {}
    for column in columns:
        rates = churn_rate_by(df, column)["churn_rate"]
        gaps[column] = rates.max() - rates.min()
    return pd.Series(gaps, name="churn_rate_gap").sort_values(ascending=False)


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
    df: pd.DataFrame,
    column: str,
    ax: Axes | None = None,
    xmax: float | None = None,
    sort_by_rate: bool = True,
) -> Axes:
    """Horizontal bars of churn rate per category, with the overall rate as a dashed line."""
    # barh draws from the bottom up, so reverse to keep the first row on top
    rates = churn_rate_by(df, column, sort_by_rate).iloc[::-1]
    overall = df[TARGET].mean()
    ax = ax or plt.subplots(figsize=(6, 0.5 * len(rates) + 1))[1]

    labels = [
        f"{value}  (n={n:,})" for value, n in zip(rates.index, rates["customers"])
    ]
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


def plot_churn_rate_gaps(gaps: pd.Series, ax: Axes | None = None) -> Axes:
    """Rank features by their churn rate gap (see `churn_rate_gaps`)."""
    # barh draws from the bottom up, so sort ascending to put the largest gap on top
    gaps = gaps.sort_values()
    ax = ax or plt.subplots(figsize=(6, 0.3 * len(gaps) + 1))[1]

    bars = ax.barh(gaps.index, gaps, color=NEUTRAL_COLOR, height=0.6)
    ax.bar_label(bars, labels=[f"{g * 100:.0f}" for g in gaps], padding=3)

    ax.set_title("How strongly each feature separates churners")
    ax.set_xlabel(
        "Highest minus lowest churn rate within the feature (percentage points)"
    )
    ax.set_xlim(0, gaps.max() * 1.12)
    ax.xaxis.set_major_formatter(FuncFormatter(lambda x, _: f"{x * 100:.0f}"))
    ax.grid(axis="y", visible=False)
    return ax


def plot_numeric_by_churn(
    df: pd.DataFrame, column: str, ax: Axes | None = None, bins: int = 30
) -> Axes:
    """Distribution of a numeric column for stayed vs churned customers.

    Each group is normalised to its own total (density), so the smaller churned
    group is comparable to the larger stayed group.
    """
    ax = ax or plt.subplots(figsize=(5, 3.5))[1]
    edges = np.histogram_bin_edges(df[column], bins=bins)

    for value, label, color in [
        (0, "Stayed", STAYED_COLOR),
        (1, "Churned", CHURNED_COLOR),
    ]:
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


# --- Model evaluation (step 5) ---


def plot_roc_pr_curves(
    y_true: pd.Series, proba: np.ndarray, threshold: float
) -> Figure:
    """ROC and precision-recall curves with their random baselines.

    The dot marks where the chosen threshold sits on each curve.
    """
    fig, (ax_roc, ax_pr) = plt.subplots(1, 2, figsize=(11, 4.5))
    curve_style = {"color": STAYED_COLOR, "linewidth": 2}
    chance_style = {"color": NEUTRAL_COLOR, "linestyle": "--", "linewidth": 1}

    RocCurveDisplay.from_predictions(
        y_true,
        proba,
        ax=ax_roc,
        name="Model",
        curve_kwargs=curve_style,
        plot_chance_level=True,
        chance_level_kw=chance_style,
    )
    PrecisionRecallDisplay.from_predictions(
        y_true,
        proba,
        ax=ax_pr,
        name="Model",
        curve_kwargs=curve_style,
        plot_chance_level=True,
        chance_level_kw=chance_style,
    )

    # where the chosen threshold sits on each curve
    y_true = np.asarray(y_true)
    predicted = proba >= threshold
    recall = predicted[y_true == 1].mean()
    false_positive_rate = predicted[y_true == 0].mean()
    precision = y_true[predicted].mean()
    for ax, x, y in [
        (ax_roc, false_positive_rate, recall),
        (ax_pr, recall, precision),
    ]:
        ax.plot(
            x,
            y,
            "o",
            color=CHURNED_COLOR,
            markersize=8,
            label=f"threshold {threshold:.2f}",
        )
        ax.legend(loc="lower right" if ax is ax_roc else "upper right")

    ax_roc.set_title("ROC curve")
    ax_roc.set_xlabel("False positive rate (share of stayers flagged)")
    ax_roc.set_ylabel("True positive rate (recall)")
    ax_pr.set_title("Precision-recall curve")
    ax_pr.set_xlabel("Recall (share of churners caught)")
    ax_pr.set_ylabel("Precision (share of flagged who churn)")
    fig.tight_layout()
    return fig


def plot_confusion_matrix(
    y_true: pd.Series, proba: np.ndarray, threshold: float, ax: Axes | None = None
) -> Axes:
    """Confusion matrix of the predictions at `threshold`."""
    display = ConfusionMatrixDisplay.from_predictions(
        y_true,
        (proba >= threshold).astype(int),
        display_labels=["Stayed", "Churned"],
        cmap="Blues",
        colorbar=False,
        ax=ax,
    )
    display.ax_.set_title(f"Confusion matrix at threshold {threshold:.2f}")
    display.ax_.set_xlabel("Predicted")
    display.ax_.set_ylabel("Actual")
    display.ax_.grid(False)
    return display.ax_


def plot_threshold_tradeoff(
    table: pd.DataFrame, threshold: float, ax: Axes | None = None
) -> Axes:
    """Precision, recall and share of customers flagged across thresholds."""
    # skip thresholds that flag almost nobody: precision gets noisy there
    table = table[table["flagged_share"] >= 0.01]
    ax = ax or plt.subplots(figsize=(7, 4))[1]

    ax.plot(
        table.index, table["precision"], color=STAYED_COLOR, lw=2, label="Precision"
    )
    ax.plot(table.index, table["recall"], color=CHURNED_COLOR, lw=2, label="Recall")
    ax.plot(
        table.index,
        table["flagged_share"],
        color=NEUTRAL_COLOR,
        lw=2,
        linestyle="--",
        label="Customers flagged",
    )
    ax.axvline(0.5, color=NEUTRAL_COLOR, linestyle=":", linewidth=1)
    ax.axvline(threshold, color=TEXT_COLOR, linestyle="--", linewidth=1)
    ax.annotate(
        f"chosen {threshold:.2f}",
        (threshold, 1.0),
        xytext=(-4, 0),
        textcoords="offset points",
        ha="right",
        va="top",
    )
    ax.annotate(
        "default 0.5",
        (0.5, 1.0),
        xytext=(4, 0),
        textcoords="offset points",
        va="top",
        color=MUTED_COLOR,
    )

    ax.set_title("Choosing the decision threshold")
    ax.set_xlabel("Threshold (flag a customer if churn probability ≥ threshold)")
    ax.set_ylim(0, 1.05)
    ax.yaxis.set_major_formatter(PercentFormatter(1.0))
    ax.legend(loc="center right")
    return ax


def plot_feature_importance(importances: pd.DataFrame, ax: Axes | None = None) -> Axes:
    """Horizontal bars of permutation importance, with the std as error bars."""
    # barh draws from the bottom up, so reverse to put the largest on top
    importances = importances.iloc[::-1]
    ax = ax or plt.subplots(figsize=(7, 0.3 * len(importances) + 1))[1]

    ax.barh(
        importances.index,
        importances["importance"],
        xerr=importances["std"],
        color=NEUTRAL_COLOR,
        height=0.6,
        error_kw={"ecolor": MUTED_COLOR, "elinewidth": 1},
    )
    ax.axvline(0, color=MUTED_COLOR, linewidth=0.8)
    ax.set_title("What drives the predictions? (permutation importance)")
    ax.set_xlabel("Drop in PR-AUC when the feature is shuffled")
    ax.grid(axis="y", visible=False)
    return ax


def plot_calibration(
    y_true: pd.Series, proba: np.ndarray, ax: Axes | None = None
) -> Axes:
    """Predicted probability vs actual churn rate, in 10 equal-sized groups of customers."""
    ax = ax or plt.subplots(figsize=(5, 5))[1]
    CalibrationDisplay.from_predictions(
        y_true,
        proba,
        n_bins=10,
        strategy="quantile",
        name="Model",
        ax=ax,
        color=STAYED_COLOR,
    )
    ax.set_title("Are the probabilities honest?")
    ax.set_xlabel("Average predicted churn probability")
    ax.set_ylabel("Actual churn rate")
    return ax
