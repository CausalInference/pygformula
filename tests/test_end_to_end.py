"""End-to-end tests running the full parametric g-formula pipeline.

These tests build a :class:`ParametricGformula` object on simulated long-format
data, call ``fit()`` and inspect the resulting risk estimates, fitted model
coefficients and simulated pools. Plotting is never invoked.
"""

import types
from typing import Callable, List, Optional

import numpy as np
import pandas as pd
import pytest
from statsmodels.discrete.discrete_model import MNLogit

from pygformula import ParametricGformula
from pygformula.interventions import static
from pygformula.parametric_gformula.fit import fit_covariate_model


def _fit_gformula(
    data: pd.DataFrame,
    covnames: List[str],
    covtypes: List[str],
    covmodels: List[str],
    ymodel: str,
    seed: int = 123,
    n_simul: Optional[int] = None,
) -> ParametricGformula:
    """Fit a parametric g-formula model on ``data`` without plotting.

    Args:
        data: Long-format observed data with ``id``, ``t0``, the covariates and
            the outcome column ``Y``.
        covnames: Names of the time-varying covariates.
        covtypes: Type of each covariate (e.g. ``"categorical"`` or ``"binary"``).
        covmodels: Model formula for each covariate.
        ymodel: Model formula for the outcome.
        seed: Random seed used for the simulation.
        n_simul: Number of subjects to simulate; ``None`` keeps the package
            default (the number of observed subjects).

    Returns:
        The fitted ``ParametricGformula`` object (``fit()`` already called).
    """
    time_points = int(data["t0"].max()) + 1
    g = ParametricGformula(
        obs_data=data,
        id="id",
        time_name="t0",
        outcome_name="Y",
        ymodel=ymodel,
        time_points=time_points,
        int_descript=["Never treat"],
        Intervention1_A=[static, np.zeros(time_points)],
        covnames=covnames,
        covtypes=covtypes,
        covmodels=covmodels,
        outcome_type="survival",
        nsamples=0,
        model_fits=True,
        seed=seed,
        **({"n_simul": n_simul} if n_simul is not None else {}),
    )
    g.fit()
    return g


@pytest.mark.parametrize("K", [3, 6, 11])
def test_gformula_categorical_cov(
    make_cat_data: Callable[..., pd.DataFrame], K: int
) -> None:
    """Run the full pipeline for a categorical covariate with ``K`` levels.

    Args:
        make_cat_data: Factory fixture producing categorical Markov data.
        K: Number of categories of the ``L`` covariate.
    """
    data = make_cat_data(K)
    g = _fit_gformula(
        data,
        covnames=["L", "A"],
        covtypes=["categorical", "binary"],
        covmodels=["L ~ C(lag1_L) + t0", "A ~ C(L) + C(lag1_L) + t0"],
        ymodel="Y ~ C(lag1_L) + A",
    )

    assert np.isfinite(np.asarray(g.g_results, dtype=float)).all()

    model_coeffs = g.summary_dict["model_coeffs"]
    assert np.isfinite(np.asarray(model_coeffs["L"], dtype=float)).all()

    natural_course = g.pool_dict["Natural course"]
    observed = set(np.unique(data["L"].to_numpy()).tolist())
    simulated = set(np.unique(natural_course["L"].to_numpy()).tolist())
    assert simulated.issubset(observed)


def _drop_late_rows(data: pd.DataFrame, label: int) -> pd.DataFrame:
    """Drop the rows with ``L == label`` and ``t0 > 0``, keeping them at ``t0 == 0``."""
    return data[~((data["L"] == label) & (data["t0"] > 0))]


def test_gformula_category_only_at_time_zero(
    make_cat_data: Callable[..., pd.DataFrame],
) -> None:
    """A category present only at ``t0 == 0`` is never simulated and keeps its frequency.

    Args:
        make_cat_data: Factory fixture producing categorical Markov data.
    """
    data = make_cat_data(6)
    data["L"] = pd.Categorical(data["L"])
    data = _drop_late_rows(data, 2)

    g = _fit_gformula(
        data,
        covnames=["L", "A"],
        covtypes=["categorical", "binary"],
        covmodels=["L ~ C(lag1_L) + t0", "A ~ C(L) + C(lag1_L) + t0"],
        ymodel="Y ~ C(lag1_L) + A",
    )

    # ``g.fit()`` already runs ``comparison_calculate`` (comparisons.py); reaching
    # this point means that comparison step did not crash.
    natural_course = g.pool_dict["Natural course"]
    later = natural_course[natural_course["t0"] > 0]
    assert not (later["L"] == 2).any()

    observed_later = data[data["t0"] > 0]
    observed_labels = pd.unique(observed_later["L"].dropna())
    for label in observed_labels:
        sim_freq = (later["L"] == label).mean()
        obs_freq = (observed_later["L"] == label).mean()
        assert abs(sim_freq - obs_freq) <= 0.05


def test_gformula_categorical_with_nan_newton(
    make_cat_data: Callable[..., pd.DataFrame], monkeypatch: pytest.MonkeyPatch
) -> None:
    """The pipeline survives a Newton step that returns NaN with ``converged=True``.

    Args:
        make_cat_data: Factory fixture producing categorical Markov data.
        monkeypatch: Pytest fixture for patching ``MNLogit.fit``.
    """
    data = make_cat_data(6)

    real_fit = MNLogit.fit

    def fake_fit(self, *args, **kwargs):
        if kwargs.get("method") == "newton":
            return types.SimpleNamespace(
                params=np.full((1, 1), np.nan),
                llf=float("nan"),
                bse=np.full((1, 1), np.nan),
                mle_retvals={"converged": True},
            )
        return real_fit(self, *args, **kwargs)

    monkeypatch.setattr(MNLogit, "fit", fake_fit)
    with pytest.warns(UserWarning, match="bfgs"):
        g = _fit_gformula(
            data,
            covnames=["L", "A"],
            covtypes=["categorical", "binary"],
            covmodels=["L ~ C(lag1_L) + t0", "A ~ C(L) + C(lag1_L) + t0"],
            ymodel="Y ~ C(lag1_L) + A",
        )

    assert np.isfinite(np.asarray(g.g_results, dtype=float)).all()


def test_gformula_spline_in_ymodel_and_covmodel() -> None:
    """A spline in both the covariate and outcome models runs end to end.

    ``L2`` is a continuous covariate generated from an AR(1) process whose
    baseline (``t0 == 0``) values are drawn from a wider distribution than the
    later values. This keeps every simulated ``L2`` (and hence every
    ``lag1_L2``) inside the range the ``bs(...)`` splines were fit on, because
    ``predict`` is strict about out-of-range spline inputs.
    """
    rng = np.random.default_rng(0)
    n_ids = 1500
    n_times = 5

    L2 = np.zeros((n_ids, n_times))
    A = np.zeros((n_ids, n_times), dtype=int)

    L2[:, 0] = rng.normal(0.0, 2.0, size=n_ids)
    A[:, 0] = rng.binomial(1, 0.4, size=n_ids)

    for t in range(1, n_times):
        L2[:, t] = (
            0.5 * L2[:, t - 1] + 0.3 * A[:, t - 1] + rng.normal(0.0, 0.5, size=n_ids)
        )
        logit_a = -0.5 + 1.5 * A[:, t - 1] + 0.3 * L2[:, t]
        A[:, t] = rng.binomial(1, 1.0 / (1.0 + np.exp(-logit_a)))

    lag_L2 = np.zeros_like(L2)
    lag_L2[:, 1:] = L2[:, :-1]
    lag_A = np.zeros_like(A)
    lag_A[:, 1:] = A[:, :-1]

    logit_y = -3.0 + 0.5 * L2 + 0.5 * A
    Y = rng.binomial(1, 1.0 / (1.0 + np.exp(-logit_y)))

    data = pd.DataFrame(
        {
            "id": np.repeat(np.arange(n_ids), n_times),
            "t0": np.tile(np.arange(n_times), n_ids),
            "L2": L2.ravel(),
            "lag1_L2": lag_L2.ravel(),
            "A": A.ravel(),
            "lag1_A": lag_A.ravel(),
            "Y": Y.ravel(),
        }
    )

    g = _fit_gformula(
        data,
        covnames=["L2", "A"],
        covtypes=["normal", "binary"],
        covmodels=["L2 ~ bs(lag1_L2, df=4) + lag1_A + t0", "A ~ lag1_A + L2 + t0"],
        ymodel="Y ~ bs(L2, df=4) + A + t0",
    )

    results = np.asarray(g.g_results, dtype=float)
    assert np.isfinite(results).all()
    assert (results >= 0.0).all() and (results <= 1.0).all()


def _make_quasi_separated_data() -> pd.DataFrame:
    """Return K=6 categorical data with an empty ``(lag1_L, L)`` cell.

    Mirrors the data of ``test_multinomial_quasi_separation_contract``: the
    outcome is nearly identical to its lag (3% of rows flip to the next label),
    so some ``(lag1_L, L)`` cells are empty, and ``x`` is an unscaled predictor
    in ``[18, 89]``. This makes the multinomial Hessian non-invertible.
    """
    rng = np.random.default_rng(7)
    n = 4000
    lag = rng.integers(0, 6, size=n)
    outcome = lag.copy()
    flip = rng.random(n) < 0.03
    outcome[flip] = (lag[flip] + 1) % 6
    x = rng.uniform(18.0, 89.0, size=n)
    data = pd.DataFrame({"L": outcome, "lag1_L": lag, "x": x})
    data["t0"] = 1
    return data


def test_fit_covariate_model_return_fits_without_covariance() -> None:
    """``return_fits=True`` stores ``summary()`` even without a covariance matrix.

    The accepted multinomial fit has no covariance matrix (quasi-separated
    data), so ``cov_params()`` is NaN and the summary entry for ``L`` must still
    be produced.
    """
    data = _make_quasi_separated_data()

    with pytest.warns(UserWarning):
        (
            covariate_fits,
            _bounds,
            _rmses,
            model_coeffs,
            model_stderrs,
            model_vcovs,
            model_fits_summary,
        ) = fit_covariate_model(
            covmodels=["L ~ C(lag1_L) + x"],
            covnames=["L"],
            covtypes=["categorical"],
            covfits_custom=["NA"],
            time_name="t0",
            obs_data=data,
            return_fits=True,
        )

    fit = covariate_fits["L"]
    cov_params = np.asarray(fit.cov_params(), dtype=float)
    assert np.isnan(cov_params).all()

    coeffs = np.asarray(model_coeffs["L"], dtype=float)
    assert np.isfinite(coeffs).all()

    stderrs = np.asarray(model_stderrs["L"], dtype=float)
    assert stderrs.shape == coeffs.shape
    assert np.isnan(stderrs).all()

    vcovs = np.asarray(model_vcovs["L"], dtype=float)
    assert vcovs.shape == (coeffs.size, coeffs.size)
    assert np.isnan(vcovs).all()

    assert "L" in model_fits_summary


def test_gformula_categorical_recovers_known_markov_chain() -> None:
    """The categorical simulation recovers a known first-order Markov chain.

    Builds data whose covariate ``L`` follows a known, time-homogeneous Markov
    chain with a fixed transition matrix and initial distribution, fits the
    parametric g-formula, then checks that the simulated ``Natural course`` pool
    reproduces both the per-step transition probabilities and the marginal
    distribution at every later time point. ``A`` and ``Y`` are independent
    noise, so any mismatch points at the categorical simulation itself rather
    than at model misspecification or attrition.
    """
    rng = np.random.default_rng(42)
    n_ids = 3000
    n_times = 5
    n_states = 4

    true_trans = np.array(
        [
            [0.70, 0.15, 0.10, 0.05],
            [0.10, 0.70, 0.15, 0.05],
            [0.05, 0.10, 0.70, 0.15],
            [0.15, 0.05, 0.10, 0.70],
        ]
    )
    initial = np.array([0.4, 0.3, 0.2, 0.1])

    levels = np.zeros((n_ids, n_times), dtype=int)
    levels[:, 0] = rng.choice(n_states, size=n_ids, p=initial)
    cum = true_trans.cumsum(axis=1)
    for t in range(1, n_times):
        u = rng.random(n_ids)
        idx = (u[:, None] > cum[levels[:, t - 1]]).sum(axis=1)
        levels[:, t] = np.minimum(idx, n_states - 1)

    lag = np.zeros_like(levels)
    lag[:, 1:] = levels[:, :-1]

    n_rows = n_ids * n_times
    data = pd.DataFrame(
        {
            "id": np.repeat(np.arange(n_ids), n_times),
            "t0": np.tile(np.arange(n_times), n_ids),
            "L": levels.ravel(),
            "lag1_L": lag.ravel(),
            "A": rng.binomial(1, 0.5, n_rows),
            "Y": rng.binomial(1, 0.05, n_rows),
        }
    )

    g = _fit_gformula(
        data,
        covnames=["L", "A"],
        covtypes=["categorical", "binary"],
        covmodels=["L ~ C(lag1_L) + t0", "A ~ C(L) + C(lag1_L) + t0"],
        ymodel="Y ~ C(lag1_L) + A",
        n_simul=10000,
    )

    assert np.isfinite(np.asarray(g.g_results, dtype=float)).all()

    natural_course = g.pool_dict["Natural course"].sort_values(["id", "t0"]).reset_index(
        drop=True
    )
    natural_course["lag_L_sim"] = natural_course.groupby("id")["L"].shift(1)
    later = natural_course[natural_course["t0"] > 0]

    observed_later = data[data["t0"] > 0]
    n_fit_from = np.array([(observed_later["lag1_L"] == s).sum() for s in range(n_states)])

    sim_L = later["L"].to_numpy()
    sim_lag = later["lag_L_sim"].to_numpy()
    n_sim_from = np.array([(sim_lag == s).sum() for s in range(n_states)])

    # Check 1: row-normalised simulated transition table vs the true matrix.
    for origin in range(n_states):
        for dest in range(n_states):
            p = true_trans[origin, dest]
            sim = ((sim_lag == origin) & (sim_L == dest)).sum() / n_sim_from[origin]
            tol = (
                4.0
                * np.sqrt(
                    p * (1.0 - p) * (1.0 / n_fit_from[origin] + 1.0 / n_sim_from[origin])
                )
                + 0.005
            )
            assert abs(sim - p) <= tol, (
                f"transition origin={origin} dest={dest}: true={p:.6f} "
                f"simulated={sim:.6f} tolerance={tol:.6f}"
            )

    # Check 2: simulated marginal distribution at each t0 vs pi0 @ P^t0.
    for t0 in range(1, n_times):
        theoretical = initial @ np.linalg.matrix_power(true_trans, t0)
        at_t = natural_course[natural_course["t0"] == t0]
        n_sim_t = len(at_t)
        at_t_L = at_t["L"].to_numpy()
        for state in range(n_states):
            p = theoretical[state]
            sim = (at_t_L == state).mean()
            tol = 4.0 * np.sqrt(p * (1.0 - p) / n_sim_t) + 0.01
            assert abs(sim - p) <= tol, (
                f"marginal t0={t0} state={state}: true={p:.6f} "
                f"simulated={sim:.6f} tolerance={tol:.6f}"
            )
