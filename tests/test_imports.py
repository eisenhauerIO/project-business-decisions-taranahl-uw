"""Smoke tests to verify all modules can be imported."""


def test_import_tables():
    """Test that tables module can be imported with expected functions."""
    from auxiliary import tables

    assert hasattr(tables, "create_table1")
    assert hasattr(tables, "create_market_summary")
    assert hasattr(tables, "create_simulation_summary")


def test_import_predictions():
    """Test that predictions module re-exports simulator functions."""
    from auxiliary import predictions

    assert hasattr(predictions, "prepare_data")
    assert hasattr(predictions, "simulate_paths")
    assert hasattr(predictions, "estimate_volatility")


def test_import_plots():
    """Test that plots module can be imported with expected functions."""
    from auxiliary import plots

    assert hasattr(plots, "plot_figure1")
    assert hasattr(plots, "plot_fan_chart")
    assert hasattr(plots, "plot_probability_path")


def test_import_kalshi():
    """Test that kalshi module can be imported with expected functions."""
    from auxiliary import kalshi

    assert hasattr(kalshi, "kalshi_get")
    assert hasattr(kalshi, "get_nba_markets")
    assert hasattr(kalshi, "get_market_trades")


def test_import_simulator():
    """Test that simulator module can be imported with expected functions."""
    from auxiliary import simulator

    assert hasattr(simulator, "simulate_paths")
    assert hasattr(simulator, "bootstrap_paths")
    assert hasattr(simulator, "estimate_volatility")
