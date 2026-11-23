"""Categorical covariate where the original pygformula diverges to NaN.

This example generates a synthetic dataset in code (no download, no real data)
with a fixed random seed and fits the parametric g-formula for a categorical
time-varying covariate ``L``.

The data has 2,000 individuals over 5 time points.  ``L`` has K=6 categories
and follows near-deterministic transitions: at each time step it keeps its
previous value with 97% probability and otherwise moves to the next label, so
most ``(lag1_L, L)`` cells are empty.  An unscaled baseline predictor ``x`` in
``[18, 89]`` is added to the ``L`` model (as a ``basecovs`` entry).

This is a quasi-separated multinomial problem.  The ORIGINAL pygformula fits the
``L`` model with statsmodels' default Newton optimizer, which diverges to NaN
parameters (statsmodels still prints "Optimization terminated successfully"
with a NaN function value); the later ``simulate()`` step then raises
``ValueError: probabilities contain NaN``.  The refactored package detects the
non-finite fit, falls back to BFGS and completes, emitting a ``UserWarning``
about the fallback.
"""

import numpy as np
import pandas as pd

from pygformula import ParametricGformula
from pygformula.interventions import static

# --- Synthetic data -------------------------------------------------------
SEED = 7
N_IDS = 2000
N_TIMES = 5
K = 6  # number of categories of L

rng = np.random.default_rng(SEED)

# Unscaled baseline predictor, constant within each individual.
x = rng.uniform(18.0, 89.0, size=N_IDS)

levels = np.empty((N_IDS, N_TIMES), dtype=int)
levels[:, 0] = rng.integers(0, K, size=N_IDS)
for t in range(1, N_TIMES):
    levels[:, t] = levels[:, t - 1].copy()
    flip = rng.random(N_IDS) < 0.03
    levels[flip, t] = (levels[flip, t - 1] + 1) % K

A = rng.binomial(1, 0.4, size=(N_IDS, N_TIMES))
Y = rng.binomial(1, 0.05, size=(N_IDS, N_TIMES))

obs_data = pd.DataFrame(
    {
        "id": np.repeat(np.arange(N_IDS), N_TIMES),
        "t0": np.tile(np.arange(N_TIMES), N_IDS),
        "L": levels.ravel(),
        "A": A.ravel(),
        "Y": Y.ravel(),
        "x": np.repeat(x, N_TIMES),
    }
)

# --- Model specification ---------------------------------------------------
time_name = "t0"
id = "id"

covnames = ["L", "A"]
covtypes = ["categorical", "binary"]
covmodels = [
    "L ~ C(lag1_L) + t0 + x",
    "A ~ C(L) + C(lag1_L) + t0",
]

basecovs = ["x"]

outcome_name = "Y"
ymodel = "Y ~ C(lag1_L) + A"

time_points = int(np.max(np.unique(obs_data[time_name]))) + 1
int_descript = ["Never treat", "Always treat"]

g = ParametricGformula(
    obs_data=obs_data,
    id=id,
    time_name=time_name,
    time_points=time_points,
    int_descript=int_descript,
    Intervention1_A=[static, np.zeros(time_points)],
    Intervention2_A=[static, np.ones(time_points)],
    covnames=covnames,
    covtypes=covtypes,
    covmodels=covmodels,
    basecovs=basecovs,
    outcome_name=outcome_name,
    ymodel=ymodel,
    outcome_type="survival",
    nsamples=0,
    seed=SEED,
)
g.fit()
