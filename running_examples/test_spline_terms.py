"""Spline and categorical terms in the model formulas.

This example generates a synthetic dataset in code (no download, no real data)
with a fixed random seed and fits the parametric g-formula for a continuous
time-varying covariate ``L2`` whose model uses a B-spline ``bs(...)`` term, with
a binary lag ``lag1_A`` included as a categorical ``C(...)`` term.

The data has 1,500 individuals over 5 time points.  ``L2`` follows an AR(1)
process whose baseline (``t0 == 0``) values are drawn from a wider distribution
than the later values, which keeps every simulated ``L2`` (and hence every
``lag1_L2``) inside the range the splines were fit on.

The ORIGINAL pygformula could not fit ``bs(...)`` terms in a covariate model;
the refactored package can.  The ``A`` covariate model is deliberately written
with double spaces around its ``+`` signs, and the spline is used both in the
``L2`` covariate model and in the outcome model.
"""

import numpy as np
import pandas as pd

from pygformula import ParametricGformula
from pygformula.interventions import static

# --- Synthetic data -------------------------------------------------------
SEED = 0
N_IDS = 1500
N_TIMES = 5

rng = np.random.default_rng(SEED)

L2 = np.zeros((N_IDS, N_TIMES))
A = np.zeros((N_IDS, N_TIMES), dtype=int)

# Wider baseline distribution: keeps simulated L2 inside the bs(...) range.
L2[:, 0] = rng.normal(0.0, 2.0, size=N_IDS)
A[:, 0] = rng.binomial(1, 0.4, size=N_IDS)

for t in range(1, N_TIMES):
    L2[:, t] = (
        0.5 * L2[:, t - 1] + 0.3 * A[:, t - 1] + rng.normal(0.0, 0.5, size=N_IDS)
    )
    logit_a = -0.5 + 1.5 * A[:, t - 1] + 0.3 * L2[:, t]
    A[:, t] = rng.binomial(1, 1.0 / (1.0 + np.exp(-logit_a)))

lag_L2 = np.zeros_like(L2)
lag_L2[:, 1:] = L2[:, :-1]
lag_A = np.zeros_like(A)
lag_A[:, 1:] = A[:, :-1]

logit_y = -3.0 + 0.5 * L2 + 0.5 * A
Y = rng.binomial(1, 1.0 / (1.0 + np.exp(-logit_y)))

obs_data = pd.DataFrame(
    {
        "id": np.repeat(np.arange(N_IDS), N_TIMES),
        "t0": np.tile(np.arange(N_TIMES), N_IDS),
        "L2": L2.ravel(),
        "lag1_L2": lag_L2.ravel(),
        "A": A.ravel(),
        "lag1_A": lag_A.ravel(),
        "Y": Y.ravel(),
    }
)

# --- Model specification ---------------------------------------------------
time_name = "t0"
id = "id"

covnames = ["L2", "A"]
covtypes = ["normal", "binary"]
covmodels = [
    "L2 ~ bs(lag1_L2, df=4) + C(lag1_A) + t0",
    "A ~  lag1_A  +  L2  +  t0",
]

outcome_name = "Y"
ymodel = "Y ~ bs(L2, df=4) + A + t0"
outcome_type = "survival"

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
    outcome_name=outcome_name,
    ymodel=ymodel,
    outcome_type=outcome_type,
    nsamples=0,
    seed=SEED,
)
g.fit()
