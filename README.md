[![Review Assignment Due Date](https://classroom.github.com/assets/deadline-readme-button-22041afd0340ce965d47ae6ef1cefeee28c7c493a6346c4f15d667ab976d596c.svg)](https://classroom.github.com/a/g5Rk6CRe)
[![Run Notebook](https://github.com/eisenhauerIO/projects-businss-decisions/actions/workflows/run-notebook.yml/badge.svg)](https://github.com/eisenhauerIO/projects-businss-decisions/actions/workflows/run-notebook.yml)
[![Ruff](https://img.shields.io/endpoint?url=https://raw.githubusercontent.com/astral-sh/ruff/main/assets/badge/v2.json)](https://github.com/astral-sh/ruff)

## Project — NBA Prediction Market Simulator

**Main notebook: [project.ipynb](project.ipynb)**

This project builds an end-to-end prediction market simulator for NBA game outcome markets using live data from the [Kalshi](https://kalshi.com) prediction market exchange. It combines stochastic modelling, Monte Carlo simulation, statistical calibration, machine learning, and cross-platform market analysis.

### What the project does

- **Live market data** — Fetches real NBA game winner markets from the Kalshi API (KXNBAGAME series), including order book snapshots and trade histories
- **Logit-scale random walk** — Models probability paths as a bounded random walk in logit space (`θ = logit(p)`) and fits the volatility parameter `σ` from 50 historical games
- **Monte Carlo fan chart** — Simulates 500 forward paths from the current market price, displaying 10th/25th/75th/90th percentile bands to visualise uncertainty
- **Bootstrap simulation** — Resamples empirical logit increments from historical games as a non-parametric alternative to the parametric model
- **Calibration analysis** — Tests whether opening market prices accurately predict game outcomes (are 70% markets right 70% of the time?)
- **ML outcome prediction** — Trains a logistic regression model with leave-one-out cross-validation to predict game outcomes from market microstructure features (opening probability, logit volatility, trade count, market duration), comparing accuracy to a naive baseline
- **Cross-platform comparison** — Loads a Kaggle snapshot of Polymarket (1,044 sports moneyline markets) and compares implied probability distributions against Kalshi NBA markets

### Repository structure

| Path | Description |
|------|-------------|
| `project.ipynb` | Main project notebook (start here) |
| `auxiliary/kalshi.py` | Kalshi API client |
| `auxiliary/simulator.py` | Logit random walk, Monte Carlo, bootstrap, calibration, ML dataset builder |
| `auxiliary/plots.py` | Fan chart and calibration plot functions |
| `auxiliary/tables.py` | Summary table builders |
| `auxiliary/polymarket.py` | Polymarket dataset loader and filter |
| `tests/` | Unit tests for all auxiliary modules |

### Data

Live Kalshi data is fetched at runtime via the public API (no key required). The Polymarket dataset (`data/polymarket_markets.csv`, 202 MB) is not included in this repository — download it from [Kaggle](https://www.kaggle.com/datasets/ismetsemedov/polymarket-prediction-markets) and place it in the `data/` folder to run Section 3.5 interactively. Pre-computed outputs are embedded in the notebook for offline viewing.
