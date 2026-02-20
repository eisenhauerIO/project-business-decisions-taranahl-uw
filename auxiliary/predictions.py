"""
Re-exports from simulator.py for backward compatibility.
All simulation logic lives in auxiliary/simulator.py.
"""

from auxiliary.simulator import (  # noqa: F401
    bootstrap_paths,
    calibration_data,
    estimate_volatility,
    prepare_data,
    resample_to_grid,
    simulate_paths,
)
