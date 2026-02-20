"""Kalshi API client for pulling prediction market data."""

import time

import pandas as pd
import requests

BASE_URL = "https://api.elections.kalshi.com/trade-api/v2"

# NBA series tickers available on Kalshi
NBA_SERIES = {
    "KXNBAGAME": "NBA Game Winner",
    "KXNBA1HSPREAD": "NBA 1st Half Spread",
    "KXNBA3PT": "NBA Player 3-Pointers",
    "KXNBA2D": "NBA Double-Double",
    "KXNBAWINS": "NBA Season Wins",
}


def kalshi_get(endpoint, params=None, timeout=10):
    """
    Generic GET request to the Kalshi public API.

    Parameters
    ----------
    endpoint : str
        API path relative to BASE_URL (e.g. '/markets').
    params : dict, optional
        Query parameters.
    timeout : int
        Request timeout in seconds.

    Returns
    -------
    dict or None
        Parsed JSON response, or None on error.
    """
    url = f"{BASE_URL}/{endpoint.lstrip('/')}"
    try:
        response = requests.get(url, params=params, timeout=timeout)
        if response.status_code == 200:
            return response.json()
        return None
    except requests.RequestException:
        return None


def get_nba_markets(series_ticker="KXNBAGAME", status="open", limit=200):
    """
    Fetch NBA markets for a given series and status.

    Parameters
    ----------
    series_ticker : str
        One of the NBA_SERIES keys.
    status : str
        'open', 'closed', or 'settled'.
    limit : int
        Max number of markets to return.

    Returns
    -------
    pd.DataFrame
        DataFrame of markets with columns: ticker, title, subtitle,
        last_price, volume, close_time, event_ticker.
    """
    all_markets = []
    cursor = None

    while len(all_markets) < limit:
        params = {
            "series_ticker": series_ticker,
            "status": status,
            "limit": min(200, limit - len(all_markets)),
        }
        if cursor:
            params["cursor"] = cursor

        data = kalshi_get("/markets", params=params)
        if not data or not data.get("markets"):
            break

        all_markets.extend(data["markets"])
        cursor = data.get("cursor")
        if not cursor:
            break
        time.sleep(0.2)

    if not all_markets:
        return pd.DataFrame()

    df = pd.DataFrame(all_markets)

    # Parse close_time
    if "close_time" in df.columns:
        df["close_time"] = pd.to_datetime(df["close_time"], utc=True, errors="coerce")

    return df


def get_market_trades(ticker, limit=5000):
    """
    Pull all available trades for a single market.

    Parameters
    ----------
    ticker : str
        Kalshi market ticker (e.g. 'KXNBAGAME-26FEB10INDNYK-NYK').
    limit : int
        Max number of trades to pull.

    Returns
    -------
    pd.DataFrame or None
        Sorted DataFrame of trades with columns including:
        datetime, yes_price, probability, count, taker_side.
        Returns None if no trades found.
    """
    all_trades = []
    cursor = None

    for _ in range(20):
        params = {"limit": min(1000, limit - len(all_trades)), "ticker": ticker}
        if cursor:
            params["cursor"] = cursor

        data = kalshi_get("/markets/trades", params=params)
        if not data or not data.get("trades"):
            break

        all_trades.extend(data["trades"])
        cursor = data.get("cursor")
        if not cursor or len(all_trades) >= limit:
            break
        time.sleep(0.15)

    if not all_trades:
        return None

    df = pd.DataFrame(all_trades)
    df["datetime"] = pd.to_datetime(df["created_time"], utc=True, errors="coerce")
    df = df.sort_values("datetime").reset_index(drop=True)

    # yes_price is in cents (1–99); convert to probability (0–1)
    if "yes_price" in df.columns:
        df["probability"] = df["yes_price"] / 100

    return df


def get_settled_games(series_ticker="KXNBAGAME", n_games=50):
    """
    Fetch settled (completed) NBA markets for a series.

    Parameters
    ----------
    series_ticker : str
        NBA series ticker.
    n_games : int
        How many settled games to fetch.

    Returns
    -------
    pd.DataFrame
        Settled markets with result column.
    """
    return get_nba_markets(series_ticker=series_ticker, status="settled", limit=n_games)


def build_price_histories(tickers, limit_per_market=5000, delay=0.2):
    """
    Pull trade-based price history for a list of market tickers.

    Parameters
    ----------
    tickers : list of str
        Market tickers to fetch.
    limit_per_market : int
        Max trades per market.
    delay : float
        Seconds to wait between requests.

    Returns
    -------
    dict
        Mapping of {ticker: pd.DataFrame} for markets with trade data.
    """
    histories = {}
    for ticker in tickers:
        df = get_market_trades(ticker, limit=limit_per_market)
        if df is not None and len(df) > 1:
            histories[ticker] = df
        time.sleep(delay)
    return histories
