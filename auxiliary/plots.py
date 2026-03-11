"""Plotting functions for the NBA prediction market simulator."""

import matplotlib.dates as mdates
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns

sns.set_theme(style="whitegrid")
BLUE = "#2471A3"
RED = "#C0392B"
GRAY = "#7F8C8D"


# ---------------------------------------------------------------------------
# Single-market probability path
# ---------------------------------------------------------------------------


def plot_probability_path(trade_df, ticker="", ax=None):
    """
    Plot the probability path for a single market derived from its trades.

    Parameters
    ----------
    trade_df : pd.DataFrame
        Trades DataFrame with 'datetime' and 'probability' columns.
    ticker : str
        Market ticker used as plot title.
    ax : matplotlib.axes.Axes, optional
        Axes to draw on; creates a new figure if None.

    Returns
    -------
    matplotlib.axes.Axes
    """
    standalone = ax is None
    if standalone:
        fig, ax = plt.subplots(figsize=(12, 4))

    df = trade_df.dropna(subset=["datetime", "probability"]).sort_values("datetime")

    ax.scatter(
        df["datetime"],
        df["probability"] * 100,
        alpha=0.3,
        s=8,
        color=BLUE,
        label="Trades",
    )

    if len(df) >= 10:
        smoothed = df["probability"].rolling(20, min_periods=1).median() * 100
        ax.plot(
            df["datetime"], smoothed, color=RED, linewidth=2, label="Rolling median"
        )

    ax.axhline(50, linestyle="--", color=GRAY, alpha=0.5, linewidth=1)
    ax.set_ylim(0, 100)
    ax.set_ylabel("Yes Probability (%)")
    ax.set_xlabel("Time (UTC)")
    short_ticker = ticker.split("-")[-1] if "-" in ticker else ticker
    ax.set_title(f"Probability Path — {short_ticker}", fontsize=12)
    ax.xaxis.set_major_formatter(mdates.DateFormatter("%m/%d\n%H:%M"))
    ax.legend(fontsize=9)

    if standalone:
        plt.tight_layout()
        plt.show()
    return ax


# ---------------------------------------------------------------------------
# Fan chart (simulation output)
# ---------------------------------------------------------------------------


def plot_fan_chart(
    sim_paths, actual_path=None, p0=None, title="", quantiles=(0.10, 0.25, 0.75, 0.90)
):
    """
    Plot a fan chart of simulated probability paths with percentile bands.

    Parameters
    ----------
    sim_paths : np.ndarray, shape (n_sims, n_steps + 1)
        Output of simulator.simulate_paths or simulator.bootstrap_paths.
    actual_path : np.ndarray or None
        Actual observed probability path to overlay (same length as sim axis 1).
    p0 : float or None
        Opening probability shown as annotation.
    title : str
        Chart title.
    quantiles : tuple
        Lower/upper quantile pairs to shade (inner and outer bands).

    Returns
    -------
    matplotlib.figure.Figure
    """
    n_steps = sim_paths.shape[1] - 1
    t = np.linspace(0, 100, n_steps + 1)

    fig, ax = plt.subplots(figsize=(12, 5))

    q_lo_outer = np.percentile(sim_paths, quantiles[0] * 100, axis=0) * 100
    q_hi_outer = np.percentile(sim_paths, quantiles[3] * 100, axis=0) * 100
    q_lo_inner = np.percentile(sim_paths, quantiles[1] * 100, axis=0) * 100
    q_hi_inner = np.percentile(sim_paths, quantiles[2] * 100, axis=0) * 100
    median = np.percentile(sim_paths, 50, axis=0) * 100

    ax.fill_between(
        t,
        q_lo_outer,
        q_hi_outer,
        alpha=0.15,
        color=BLUE,
        label=f"{int(quantiles[0] * 100)}–{int(quantiles[3] * 100)}th pctile",
    )
    ax.fill_between(
        t,
        q_lo_inner,
        q_hi_inner,
        alpha=0.30,
        color=BLUE,
        label=f"{int(quantiles[1] * 100)}–{int(quantiles[2] * 100)}th pctile",
    )
    ax.plot(t, median, color=BLUE, linewidth=2.5, label="Median simulation")

    for path in sim_paths[:: max(1, len(sim_paths) // 30)]:
        ax.plot(t, path * 100, color=BLUE, alpha=0.04, linewidth=0.8)

    if actual_path is not None:
        t_actual = np.linspace(0, 100, len(actual_path))
        ax.plot(
            t_actual,
            actual_path * 100,
            color=RED,
            linewidth=2.5,
            zorder=5,
            label="Observed path",
        )

    ax.axhline(50, linestyle="--", color=GRAY, alpha=0.5, linewidth=1)
    ax.set_xlim(0, 100)
    ax.set_ylim(0, 100)
    ax.set_xlabel("% of Market Lifetime Elapsed", fontsize=11)
    ax.set_ylabel("Yes Probability (%)", fontsize=11)
    ax.set_title(title or "Simulated Probability Paths (Fan Chart)", fontsize=13)
    ax.legend(fontsize=9, loc="upper left")

    if p0 is not None:
        ax.annotate(
            f"Open: {p0 * 100:.0f}%",
            xy=(0, p0 * 100),
            xytext=(5, p0 * 100 + 5),
            fontsize=9,
            color=GRAY,
        )

    plt.tight_layout()
    return fig


# ---------------------------------------------------------------------------
# Multi-market comparison (main summary figure)
# ---------------------------------------------------------------------------


def plot_figure1(histories, title="Historical NBA Game Probability Paths"):
    """
    Plot normalised probability paths for multiple historical markets.

    Each market's time axis is normalised to [0, 100]% so paths of
    different durations can be compared on the same chart.

    Parameters
    ----------
    histories : dict
        {ticker: pd.DataFrame} from kalshi.build_price_histories.
    title : str
        Chart title.

    Returns
    -------
    matplotlib.figure.Figure
    """
    fig, ax = plt.subplots(figsize=(12, 5))
    colors = plt.cm.tab20.colors

    for i, (ticker, df) in enumerate(histories.items()):
        df = df.dropna(subset=["datetime", "probability"]).sort_values("datetime")
        if len(df) < 5:
            continue

        t_start = df["datetime"].min()
        t_end = df["datetime"].max()
        duration = (t_end - t_start).total_seconds()
        if duration == 0:
            continue

        df = df.copy()
        df["time_pct"] = (df["datetime"] - t_start).dt.total_seconds() / duration * 100
        smoothed = df["probability"].rolling(20, min_periods=1).median() * 100
        label = ticker.split("-")[-1] if "-" in ticker else ticker
        ax.plot(
            df["time_pct"],
            smoothed,
            color=colors[i % len(colors)],
            linewidth=1.5,
            alpha=0.8,
            label=label,
        )

    ax.axhline(50, linestyle="--", color=GRAY, alpha=0.4, linewidth=1)
    ax.set_xlim(0, 100)
    ax.set_ylim(0, 100)
    ax.set_xlabel("% of Market Lifetime Elapsed", fontsize=11)
    ax.set_ylabel("Yes Probability (%)", fontsize=11)
    ax.set_title(title, fontsize=13)
    if len(histories) <= 12:
        ax.legend(bbox_to_anchor=(1.01, 1), loc="upper left", fontsize=8)
    plt.tight_layout()
    return fig


# ---------------------------------------------------------------------------
# Opening probability distribution
# ---------------------------------------------------------------------------


def plot_opening_distribution(markets_df, series_label="NBA Game Winner"):
    """
    Plot the distribution of opening (last_price) probabilities across markets.

    Parameters
    ----------
    markets_df : pd.DataFrame
        Markets DataFrame with a 'last_price' column (cents).
    series_label : str
        Label for the chart title.

    Returns
    -------
    matplotlib.figure.Figure
    """
    fig, ax = plt.subplots(figsize=(9, 4))

    probs = markets_df["last_price"].dropna().clip(1, 99)

    ax.hist(probs, bins=20, color=BLUE, edgecolor="white", alpha=0.85)
    ax.axvline(50, linestyle="--", color=RED, linewidth=1.5, label="50% line")
    ax.set_xlabel("Current Yes Probability (¢)", fontsize=11)
    ax.set_ylabel("Number of Markets", fontsize=11)
    ax.set_title(f"Distribution of Market Probabilities — {series_label}", fontsize=12)
    ax.legend()
    plt.tight_layout()
    return fig


# ---------------------------------------------------------------------------
# Calibration chart
# ---------------------------------------------------------------------------


def plot_calibration(cal_df, title="Market Calibration"):
    """
    Plot a calibration chart: predicted probability vs actual win rate.

    A perfectly calibrated market lies on the 45-degree diagonal.

    Parameters
    ----------
    cal_df : pd.DataFrame
        Output of simulator.calibration_data — needs columns
        predicted_prob and actual_win_rate.
    title : str
        Chart title.

    Returns
    -------
    matplotlib.figure.Figure
    """
    fig, ax = plt.subplots(figsize=(6, 6))

    valid = cal_df.dropna(subset=["predicted_prob", "actual_win_rate"])
    counts = valid.get("count", pd.Series([50] * len(valid)))

    ax.plot(
        [0, 1],
        [0, 1],
        linestyle="--",
        color=GRAY,
        linewidth=1.5,
        label="Perfect calibration",
    )
    ax.scatter(
        valid["predicted_prob"],
        valid["actual_win_rate"],
        s=counts * 0.5 + 30,
        color=BLUE,
        alpha=0.8,
        edgecolors="white",
        linewidths=0.5,
        label="Observed",
    )

    ax.set_xlim(0, 1)
    ax.set_ylim(0, 1)
    ax.set_xlabel("Predicted Probability", fontsize=11)
    ax.set_ylabel("Actual Win Rate", fontsize=11)
    ax.set_title(title, fontsize=12)
    ax.legend(fontsize=9)
    plt.tight_layout()
    return fig


# ---------------------------------------------------------------------------
# Volatility term structure
# ---------------------------------------------------------------------------


def plot_volatility_term_structure(histories, n_steps=50):
    """
    Plot how logit-scale volatility changes over the life of a market.

    Parameters
    ----------
    histories : dict
        {ticker: pd.DataFrame} from kalshi.build_price_histories.
    n_steps : int
        Time resolution.

    Returns
    -------
    matplotlib.figure.Figure
    """
    from auxiliary.simulator import prepare_data

    bin_diffs = {i: [] for i in range(n_steps)}

    for df in histories.values():
        if df is None or len(df) < 10:
            continue
        prep = prepare_data(df)
        prep["bin"] = pd.cut(
            prep["time_pct"], bins=n_steps, labels=range(n_steps), include_lowest=True
        )
        for b, grp in prep.groupby("bin", observed=True):
            diffs = np.diff(grp["logit_prob"].values)
            bin_diffs[int(b)].extend(diffs.tolist())

    x, y, sizes = [], [], []
    for b in range(n_steps):
        vals = bin_diffs[b]
        if len(vals) > 2:
            x.append(b / n_steps * 100)
            y.append(np.std(vals))
            sizes.append(len(vals))

    fig, ax = plt.subplots(figsize=(10, 4))
    ax.scatter(
        x, y, s=[s * 0.3 + 10 for s in sizes], color=BLUE, alpha=0.7, edgecolors="white"
    )
    ax.plot(x, y, color=BLUE, linewidth=1.5, alpha=0.6)
    ax.set_xlabel("% of Market Lifetime Elapsed", fontsize=11)
    ax.set_ylabel("Logit-Scale Volatility (σ)", fontsize=11)
    ax.set_title("Volatility Term Structure Across NBA Game Markets", fontsize=12)
    plt.tight_layout()
    return fig
