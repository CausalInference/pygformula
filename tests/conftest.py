"""Shared pytest configuration and fixtures for pygformula tests."""
import os
from typing import Callable

import numpy as np
import pandas as pd
import pytest

# Run the whole test session with a headless matplotlib backend. pygformula/plot.py
# leaves the backend untouched when MPLBACKEND is set, so no GUI backend is selected.
os.environ["MPLBACKEND"] = "Agg"


def _make_cat_data(
    n_categories: int, n_ids: int = 1500, n_times: int = 5, seed: int = 0
) -> pd.DataFrame:
    """Simulate long-format data with a categorical Markov covariate.

    The covariate ``L`` follows a Markov chain over ``n_categories`` integer labels
    with a strictly positive transition matrix, so every category appears at t0 > 0.

    Args:
        n_categories: Number of categories of ``L`` (labels 0 to n_categories - 1).
        n_ids: Number of individuals.
        n_times: Number of time points per individual (t0 = 0 to n_times - 1).
        seed: Seed of the random generator. The output is deterministic given the seed.

    Returns:
        A DataFrame with columns ``id``, ``t0``, ``L``, ``lag1_L``, ``A`` and ``Y``,
        sorted by ``id`` and ``t0``. ``lag1_L`` is the previous value of ``L``; at
        t0 = 0 it equals the first category (0). ``A`` is binary and ``Y`` is a rare
        binary outcome.

    Raises:
        ValueError: If some category does not appear at t0 > 0 (``n_ids`` too small).
    """
    rng = np.random.default_rng(seed)
    k = n_categories

    # Strictly positive transition matrix: every entry is at least 0.5 / k.
    trans = rng.dirichlet(np.full(k, 2.0), size=k)
    trans = 0.5 * trans + 0.5 / k
    cum = trans.cumsum(axis=1)

    levels = np.empty((n_ids, n_times), dtype=int)
    levels[:, 0] = rng.integers(0, k, size=n_ids)
    for t in range(1, n_times):
        u = rng.random(n_ids)
        idx = (u[:, None] > cum[levels[:, t - 1]]).sum(axis=1)
        levels[:, t] = np.minimum(idx, k - 1)

    lag = np.zeros_like(levels)
    lag[:, 1:] = levels[:, :-1]

    n_rows = n_ids * n_times
    data = pd.DataFrame(
        {
            "id": np.repeat(np.arange(n_ids), n_times),
            "t0": np.tile(np.arange(n_times), n_ids),
            "L": levels.ravel(),
            "lag1_L": lag.ravel(),
            "A": rng.binomial(1, 0.4, n_rows),
            "Y": rng.binomial(1, 0.05, n_rows),
        }
    )
    if data.loc[data["t0"] > 0, "L"].nunique() != k:
        raise ValueError("Some categories do not appear at t0 > 0; increase n_ids.")
    return data


@pytest.fixture
def make_cat_data() -> Callable[..., pd.DataFrame]:
    """Return a factory ``(n_categories, n_ids=1500, n_times=5, seed=0)`` for test data."""
    return _make_cat_data
