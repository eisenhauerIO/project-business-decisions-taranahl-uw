"""
Polymarket dataset loader and analysis helpers.

Loads the Kaggle snapshot of Polymarket prediction markets
(polymarket_markets.csv) and extracts sports / NBA markets for
cross-platform comparison with Kalshi.

Dataset source
--------------
https://www.kaggle.com/datasets/ismetsemedov/polymarket-prediction-markets
Snapshot date: December 3, 2025
"""

import json

import numpy as np
import pandas as pd

# NBA team names used to filter sports markets down to basketball
_NBA_KEYWORDS = [
    "NBA", "Knicks", "Lakers", "Warriors", "Celtics", "Bucks", "Nets",
    "Heat", "Bulls", "Nuggets", "Suns", "Clippers", "Mavericks", "Spurs",
    "Rockets", "Thunder", "Timberwolves", "Pacers", "Cavaliers", "Pistons",
    "Raptors", "Magic", "Hornets", "Hawks", "Wizards", "Pelicans", "Grizzlies",
    "Jazz", "Trail Blazers", "Kings", "76ers", "Sixers",
]


def load_polymarket_sports(filepath="data/polymarket_markets.csv"):
    """
    Load the Polymarket markets CSV and return cleaned sports market data.

    Filters to binary outcome (Yes/No) moneyline-style markets.  Parses
    outcomePrices to extract the Yes-side implied probability.

    Parameters
    ----------
    filepath : str
        Path to the polymarket_markets.csv file.

    Returns
    -------
    pd.DataFrame or None
        Columns: question, event_title, yes_price, volume, closed,
        sportsMarketType.  Returns None if the file is not found.
    """
    try:
        df = pd.read_csv(filepath, low_memory=False)
    except FileNotFoundError:
        return None

    # Keep only sports markets with a defined sportsMarketType
    sports = df[df["sportsMarketType"].notna()].copy()

    # Keep only binary Yes/No markets
    def _parse_outcomes(val):
        try:
            return json.loads(val)
        except Exception:
            return []

    sports["_outcomes"] = sports["outcomes"].apply(_parse_outcomes)
    binary = sports[sports["_outcomes"].apply(
        lambda x: len(x) == 2 and set(x) == {"Yes", "No"}
    )].copy()

    # Extract Yes-side price from outcomePrices JSON
    def _yes_price(row):
        try:
            prices = json.loads(row["outcomePrices"])
            outcomes = row["_outcomes"]
            idx = outcomes.index("Yes")
            return float(prices[idx])
        except Exception:
            return np.nan

    binary["yes_price"] = binary.apply(_yes_price, axis=1)
    binary = binary.dropna(subset=["yes_price"])
    binary = binary[(binary["yes_price"] > 0) & (binary["yes_price"] < 1)]

    keep = ["question", "event_title", "yes_price", "volume",
            "closed", "sportsMarketType", "lastTradePrice"]
    available = [c for c in keep if c in binary.columns]
    return binary[available].reset_index(drop=True)


def load_polymarket_nba(filepath="data/polymarket_markets.csv"):
    """
    Load Polymarket data filtered to NBA basketball markets only.

    Parameters
    ----------
    filepath : str
        Path to the polymarket_markets.csv file.

    Returns
    -------
    pd.DataFrame or None
        Same schema as load_polymarket_sports but restricted to NBA markets.
    """
    sports = load_polymarket_sports(filepath)
    if sports is None:
        return None

    pattern = "|".join(_NBA_KEYWORDS)
    mask = (
        sports["question"].str.contains(pattern, case=False, na=False)
        | sports["event_title"].str.contains(pattern, case=False, na=False)
    )
    return sports[mask].reset_index(drop=True)


def polymarket_calibration(df, n_bins=8):
    """
    Compute calibration statistics for closed Polymarket markets.

    Uses lastTradePrice (the final price before market close) as the
    predicted probability and derives the actual outcome from the
    yes_price at settlement (0 = lost, 1 = won).

    Parameters
    ----------
    df : pd.DataFrame
        Output of load_polymarket_sports or load_polymarket_nba.
    n_bins : int
        Number of probability bins.

    Returns
    -------
    pd.DataFrame
        Columns: bin_mid, predicted_prob, actual_win_rate, count.
        Compatible with plots.plot_calibration and tables.create_calibration_table.
    """
    closed = df[df["closed"] == True].copy()  # noqa: E712
    if closed.empty or "lastTradePrice" not in closed.columns:
        return pd.DataFrame()

    closed = closed.dropna(subset=["lastTradePrice", "yes_price"])
    closed["pred_prob"] = closed["lastTradePrice"].astype(float)
    # At settlement: yes_price == 1 means Yes won; yes_price == 0 means No won
    closed["won"] = (closed["yes_price"].astype(float) > 0.5).astype(int)

    # Keep only well-defined predictions
    closed = closed[
        (closed["pred_prob"] > 0.01) & (closed["pred_prob"] < 0.99)
    ]
    if len(closed) < n_bins:
        return pd.DataFrame()

    bins = np.linspace(0, 1, n_bins + 1)
    closed["bin"] = pd.cut(closed["pred_prob"], bins=bins, include_lowest=True)
    closed["bin_mid"] = closed["bin"].apply(
        lambda x: round((x.left + x.right) / 2, 2) if pd.notna(x) else np.nan
    )

    cal = (
        closed.groupby("bin_mid", observed=True)
        .agg(
            predicted_prob=("pred_prob", "mean"),
            actual_win_rate=("won", "mean"),
            count=("won", "count"),
        )
        .reset_index()
    )
    return cal
