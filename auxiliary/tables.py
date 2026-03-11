"""Table-creation functions for the NBA prediction market simulator."""

import numpy as np
import pandas as pd


def create_table1(markets_df):
    """
    Create a summary statistics table for a set of NBA markets.

    Parameters
    ----------
    markets_df : pd.DataFrame
        Markets DataFrame from kalshi.get_nba_markets.

    Returns
    -------
    pd.DataFrame
        Summary statistics with columns: count, mean, std, min, 25%, 50%,
        75%, max for key numeric fields.
    """
    numeric_cols = [
        c for c in ["last_price", "volume", "open_interest"] if c in markets_df.columns
    ]
    if not numeric_cols:
        return pd.DataFrame()

    summary = markets_df[numeric_cols].describe().T
    summary.index = [c.replace("_", " ").title() for c in summary.index]
    summary = summary.round(2)
    return summary


def create_market_summary(histories):
    """
    Build a per-market summary table from a dictionary of trade histories.

    Parameters
    ----------
    histories : dict
        {ticker: pd.DataFrame} from kalshi.build_price_histories.

    Returns
    -------
    pd.DataFrame
        One row per market with columns: ticker, n_trades, duration_min,
        open_prob, close_prob, max_prob, min_prob, volatility.
    """
    rows = []
    for ticker, df in histories.items():
        if df is None or len(df) < 2:
            continue
        df = df.sort_values("datetime")
        p = df["probability"]
        duration = (df["datetime"].max() - df["datetime"].min()).total_seconds() / 60
        logit_p = np.log(p.clip(0.01, 0.99) / (1 - p.clip(0.01, 0.99)))
        vol = float(np.std(np.diff(logit_p.values)))

        rows.append(
            {
                "ticker": ticker,
                "n_trades": len(df),
                "duration_min": round(duration, 1),
                "open_prob": round(float(p.iloc[0]), 3),
                "close_prob": round(float(p.iloc[-1]), 3),
                "max_prob": round(float(p.max()), 3),
                "min_prob": round(float(p.min()), 3),
                "logit_volatility": round(vol, 4),
            }
        )

    return pd.DataFrame(rows)


def create_calibration_table(cal_df):
    """
    Format calibration data as a display table.

    Parameters
    ----------
    cal_df : pd.DataFrame
        Output of simulator.calibration_data.

    Returns
    -------
    pd.DataFrame
        Formatted table with readable column names and rounded values.
    """
    df = cal_df.copy().dropna(subset=["predicted_prob", "actual_win_rate"])
    df = df.rename(
        columns={
            "bin_mid": "Probability Bin",
            "predicted_prob": "Avg Predicted Prob",
            "actual_win_rate": "Actual Win Rate",
            "count": "N Markets",
        }
    )
    numeric_cols = ["Avg Predicted Prob", "Actual Win Rate"]
    df[numeric_cols] = df[numeric_cols].round(3)
    return df.reset_index(drop=True)


def create_ml_summary(coef_series, baseline_acc, ml_acc):
    """
    Format logistic regression results as a display table.

    Parameters
    ----------
    coef_series : pd.Series
        Coefficients indexed by feature name.
    baseline_acc : float
        Accuracy of the naive baseline (always predict the favourite).
    ml_acc : float
        Leave-one-out cross-validated accuracy of the logistic regression.

    Returns
    -------
    pd.DataFrame
        Feature coefficients table, plus a two-row accuracy comparison.
    """
    coef_df = pd.DataFrame(
        {
            "Feature": coef_series.index,
            "Coefficient": coef_series.values.round(3),
            "Direction": [
                "↑ helps win" if c > 0 else "↓ hurts win" for c in coef_series.values
            ],
        }
    )

    accuracy_df = pd.DataFrame(
        {
            "Model": [
                "Baseline (always pick favourite)",
                "Logistic Regression (LOO-CV)",
            ],
            "Accuracy (%)": [round(baseline_acc * 100, 1), round(ml_acc * 100, 1)],
        }
    )

    return coef_df, accuracy_df


def create_simulation_summary(sim_paths, p0):
    """
    Summarise a set of simulated paths into a stats table.

    Parameters
    ----------
    sim_paths : np.ndarray, shape (n_sims, n_steps + 1)
        Output of simulator.simulate_paths.
    p0 : float
        Opening probability.

    Returns
    -------
    pd.DataFrame
        Summary at key time checkpoints (0%, 25%, 50%, 75%, 100%).
    """
    checkpoints = [0, 25, 50, 75, 100]
    n_steps = sim_paths.shape[1] - 1
    rows = []

    for cp in checkpoints:
        idx = int(cp / 100 * n_steps)
        col = sim_paths[:, idx] * 100
        rows.append(
            {
                "Time (% of market life)": cp,
                "Median (%)": round(float(np.median(col)), 1),
                "10th pctile (%)": round(float(np.percentile(col, 10)), 1),
                "90th pctile (%)": round(float(np.percentile(col, 90)), 1),
                "Prob > 50% (%)": round(float((col > 50).mean() * 100), 1),
            }
        )

    df = pd.DataFrame(rows)
    return df
