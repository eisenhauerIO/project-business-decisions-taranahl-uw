"""
Prediction market simulator using a logit-scale random walk model.

Model
-----
Kalshi market probabilities p(t) are bounded in (0, 1) and follow a
martingale under the efficient markets hypothesis.  We model them on
the logit scale:

    theta(t) = logit(p(t)) = log(p / (1 - p))
    d theta   = sigma * dW          (Brownian motion)
    p(t)      = sigmoid(theta(t))

Volatility sigma is estimated from historical trade data.
Simulation generates Monte Carlo paths that respect the [0,1] boundary.
"""

import numpy as np
import pandas as pd

# ---------------------------------------------------------------------------
# Data preparation
# ---------------------------------------------------------------------------


def prepare_data(trade_df, n_bins=100):
    """
    Normalise a raw trades DataFrame for modelling.

    Adds a 'time_pct' column (0–100) representing how far through the
    market's life each trade occurred, and a 'logit_prob' column for
    the logit-transformed probability.

    Parameters
    ----------
    trade_df : pd.DataFrame
        Output of kalshi.get_market_trades — must have 'datetime' and
        'probability' columns.
    n_bins : int
        Number of equal-width time buckets for resampling.

    Returns
    -------
    pd.DataFrame
        Prepared DataFrame with extra columns: time_pct, logit_prob,
        and a resampled median probability per bin.
    """
    df = trade_df.copy().dropna(subset=["datetime", "probability"])
    df = df.sort_values("datetime").reset_index(drop=True)

    t_start = df["datetime"].min()
    t_end = df["datetime"].max()
    duration = (t_end - t_start).total_seconds()

    if duration == 0:
        df["time_pct"] = 0.0
    else:
        df["time_pct"] = (df["datetime"] - t_start).dt.total_seconds() / duration * 100

    # Clip away exact 0/1 to avoid logit blowing up
    p = df["probability"].clip(0.01, 0.99)
    df["logit_prob"] = np.log(p / (1 - p))

    return df


def resample_to_grid(trade_df, n_steps=100):
    """
    Resample a trade history to a uniform time grid with n_steps points.

    Uses the median yes_price within each bucket.

    Parameters
    ----------
    trade_df : pd.DataFrame
        Prepared trades (output of prepare_data).
    n_steps : int
        Number of evenly-spaced time steps.

    Returns
    -------
    np.ndarray, shape (n_steps,)
        Probability at each time step (forward-filled where no trades).
    """
    df = prepare_data(trade_df, n_bins=n_steps)
    bins = np.linspace(0, 100, n_steps + 1)
    df["bin"] = pd.cut(df["time_pct"], bins=bins, labels=False, include_lowest=True)
    grid = df.groupby("bin", observed=True)["probability"].median()

    # Forward-fill gaps (no trades in that bucket)
    full_index = pd.RangeIndex(n_steps)
    grid = grid.reindex(full_index).ffill().bfill()
    return grid.values


# ---------------------------------------------------------------------------
# Volatility estimation
# ---------------------------------------------------------------------------


def estimate_volatility(histories, n_steps=100):
    """
    Estimate the logit-scale volatility (sigma) from historical paths.

    For each path we:
      1. Resample to n_steps uniform time points.
      2. Compute logit(p) at each step.
      3. Compute first differences of logit(p).
      4. Pool all differences and take the standard deviation.

    Parameters
    ----------
    histories : dict
        {ticker: pd.DataFrame} from kalshi.build_price_histories.
    n_steps : int
        Resolution for resampling each path.

    Returns
    -------
    float
        Estimated per-step standard deviation on the logit scale.
    """
    all_diffs = []

    for df in histories.values():
        if df is None or len(df) < 5:
            continue
        path = resample_to_grid(df, n_steps=n_steps)
        path = np.clip(path, 0.01, 0.99)
        logit_path = np.log(path / (1 - path))
        diffs = np.diff(logit_path)
        all_diffs.extend(diffs.tolist())

    if not all_diffs:
        return 0.15  # sensible default if no data

    return float(np.std(all_diffs))


# ---------------------------------------------------------------------------
# Simulation
# ---------------------------------------------------------------------------


def simulate_paths(p0, n_steps, sigma, n_sims=500, seed=None):
    """
    Simulate Monte Carlo probability paths using a logit random walk.

    Parameters
    ----------
    p0 : float
        Opening probability in (0, 1).
    n_steps : int
        Number of time steps to simulate.
    sigma : float
        Per-step logit-scale volatility (from estimate_volatility).
    n_sims : int
        Number of simulation paths to generate.
    seed : int, optional
        Random seed for reproducibility.

    Returns
    -------
    np.ndarray, shape (n_sims, n_steps + 1)
        Each row is one simulated probability path, starting at p0.
    """
    if seed is not None:
        rng = np.random.default_rng(seed)
    else:
        rng = np.random.default_rng()

    p0 = np.clip(p0, 0.01, 0.99)
    theta0 = np.log(p0 / (1 - p0))

    # Random walk on logit scale
    shocks = rng.normal(0, sigma, size=(n_sims, n_steps))
    theta = np.zeros((n_sims, n_steps + 1))
    theta[:, 0] = theta0
    theta[:, 1:] = theta0 + np.cumsum(shocks, axis=1)

    # Convert back to probabilities
    paths = 1 / (1 + np.exp(-theta))
    return paths


def bootstrap_paths(histories, p0, n_sims=200, n_steps=100, seed=None):
    """
    Generate simulated paths by bootstrap-resampling historical increments.

    Rather than assuming Gaussian increments, this draws directly from
    the empirical distribution of logit-scale changes.

    Parameters
    ----------
    histories : dict
        {ticker: pd.DataFrame} from kalshi.build_price_histories.
    p0 : float
        Opening probability for the simulated game.
    n_sims : int
        Number of simulation paths.
    n_steps : int
        Length of each path.
    seed : int, optional
        Random seed.

    Returns
    -------
    np.ndarray, shape (n_sims, n_steps + 1)
        Simulated probability paths.
    """
    if seed is not None:
        rng = np.random.default_rng(seed)
    else:
        rng = np.random.default_rng()

    # Collect all empirical logit increments
    all_diffs = []
    for df in histories.values():
        if df is None or len(df) < 5:
            continue
        path = resample_to_grid(df, n_steps=n_steps)
        path = np.clip(path, 0.01, 0.99)
        logit_path = np.log(path / (1 - path))
        all_diffs.extend(np.diff(logit_path).tolist())

    if not all_diffs:
        # Fall back to Gaussian with default sigma
        return simulate_paths(p0, n_steps, sigma=0.15, n_sims=n_sims, seed=seed)

    empirical = np.array(all_diffs)
    p0 = np.clip(p0, 0.01, 0.99)
    theta0 = np.log(p0 / (1 - p0))

    # Bootstrap: sample increments with replacement
    idx = rng.integers(0, len(empirical), size=(n_sims, n_steps))
    shocks = empirical[idx]

    theta = np.zeros((n_sims, n_steps + 1))
    theta[:, 0] = theta0
    theta[:, 1:] = theta0 + np.cumsum(shocks, axis=1)

    paths = 1 / (1 + np.exp(-theta))
    return paths


# ---------------------------------------------------------------------------
# Machine learning helpers
# ---------------------------------------------------------------------------


def build_ml_dataset(histories, settled_markets):
    """
    Build a labelled dataset for supervised learning from trade histories.

    Extracts market microstructure features (opening probability, logit
    volatility, trade count, market duration) from each historical game
    and joins them with the binary win/loss outcome.

    Parameters
    ----------
    histories : dict
        {ticker: pd.DataFrame} from kalshi.build_price_histories.
    settled_markets : pd.DataFrame
        Settled markets with 'ticker' and 'result' (yes/no) columns.

    Returns
    -------
    pd.DataFrame
        One row per market with columns: ticker, open_prob, logit_volatility,
        n_trades, duration_min, won (0/1).  Rows with missing outcomes are
        dropped.
    """
    result_map = (
        settled_markets.set_index("ticker")["result"]
        if "result" in settled_markets.columns
        else {}
    )

    rows = []
    for ticker, df in histories.items():
        if df is None or len(df) < 5:
            continue
        df = df.sort_values("datetime")
        p = df["probability"].clip(0.01, 0.99)
        logit_p = np.log(p / (1 - p))
        vol = float(np.std(np.diff(logit_p.values)))
        duration = (df["datetime"].max() - df["datetime"].min()).total_seconds() / 60

        result = result_map.get(ticker) if hasattr(result_map, "get") else None
        if result not in ("yes", "no"):
            continue

        rows.append(
            {
                "ticker": ticker,
                "open_prob": round(float(p.iloc[0]), 4),
                "logit_volatility": round(vol, 4),
                "n_trades": len(df),
                "duration_min": round(duration, 1),
                "won": 1 if result == "yes" else 0,
            }
        )

    return pd.DataFrame(rows)


# ---------------------------------------------------------------------------
# Calibration helpers
# ---------------------------------------------------------------------------


def calibration_data(resolved_df, n_bins=10):
    """
    Compute calibration statistics for resolved markets.

    Groups markets by their last_price before resolution and checks
    what fraction actually resolved YES in each bin.

    Parameters
    ----------
    resolved_df : pd.DataFrame
        Settled markets with 'last_price' (cents) and 'result' columns.
    n_bins : int
        Number of probability bins.

    Returns
    -------
    pd.DataFrame
        Columns: bin_mid, predicted_prob, actual_win_rate, count.
    """
    df = resolved_df.copy().dropna(subset=["last_price", "result"])

    # last_price is in cents (0–99); convert to probability
    df["pred_prob"] = df["last_price"] / 100
    df["won"] = (df["result"] == "yes").astype(int)

    bins = np.linspace(0, 1, n_bins + 1)
    df["bin"] = pd.cut(df["pred_prob"], bins=bins, include_lowest=True)
    df["bin_mid"] = df["bin"].apply(
        lambda x: round((x.left + x.right) / 2, 2) if pd.notna(x) else np.nan
    )

    cal = (
        df.groupby("bin_mid", observed=True)
        .agg(
            predicted_prob=("pred_prob", "mean"),
            actual_win_rate=("won", "mean"),
            count=("won", "count"),
        )
        .reset_index()
    )
    return cal
