"""compevent predict with C() and bs() terms returns valid probabilities."""
import numpy as np
import pandas as pd

from pygformula.parametric_gformula.fit import fit_compevent_model

COMPEVENT_MODEL = "D ~ C(g) + bs(L2, df=3) + t0"


def _make_data(n_ids: int = 300, n_times: int = 4, seed: int = 0) -> pd.DataFrame:
    """Simulate long data with a categorical ``g``, continuous ``L2`` and binary ``D``.

    Args:
        n_ids: Number of individuals.
        n_times: Number of time points per individual.
        seed: Seed of the random generator.

    Returns:
        A long-format DataFrame with columns ``id``, ``t0``, ``g``, ``L2`` and ``D``.
    """
    rng = np.random.default_rng(seed)
    g = rng.choice(["a", "b", "c"], size=n_ids)
    l2 = rng.normal(size=n_ids)
    return pd.DataFrame(
        {
            "id": np.repeat(np.arange(n_ids), n_times),
            "t0": np.tile(np.arange(n_times), n_ids),
            "g": np.repeat(g, n_times),
            "L2": np.repeat(l2, n_times),
            "D": rng.binomial(1, 0.2, n_ids * n_times),
        }
    )


def _fit(data: pd.DataFrame):
    """Fit the compevent model with the real signature of fit_compevent_model."""
    # The function returns a tuple whose first element is the fitted model.
    fit, *_ = fit_compevent_model(
        compevent_model=COMPEVENT_MODEL,
        compevent_name="D",
        time_name="t0",
        obs_data=data,
        return_fits=False,
        compevent_restrictions=None,
    )
    return fit


def test_compevent_predict_with_categorical_and_spline_terms() -> None:
    """predict(new_df) at t > 0 returns one finite probability in [0, 1] per row."""
    data = _make_data()
    fit = _fit(data)
    # Only seen levels of g and L2 inside the training range (predict is strict).
    new_df = data[data["t0"] > 0].copy()
    prob = np.asarray(fit.predict(new_df))
    assert prob.shape == (len(new_df),)
    assert np.isfinite(prob).all()
    assert ((prob >= 0) & (prob <= 1)).all()
