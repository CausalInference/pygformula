"""Tests for multinomial (categorical) covariate models.

These tests cover the multinomial wrapper and its integration in
``fit_covariate_model`` on simulated data, including the case where Newton
returns NaN parameters while reporting convergence.
"""

import types
from typing import Any, Tuple

import numpy as np
import pandas as pd
import pytest
from statsmodels.discrete.discrete_model import MNLogit

from pygformula.parametric_gformula.fit import fit_covariate_model
from pygformula.utils.multinomial import fit_multinomial_formula

FORMULA = "L ~ C(lag1_L) + t0"


def _fit(data: pd.DataFrame) -> Tuple[Any, Any, Any]:
    """Fit the categorical covariate model for ``L``.

    Args:
        data: Long-format data with columns ``L``, ``lag1_L`` and ``t0``.

    Returns:
        A tuple ``(fit, coeffs, stderrs)`` with the fitted model, its parameter
        estimates and their standard errors.
    """
    result = fit_covariate_model(
        covmodels=[FORMULA],
        covnames=["L"],
        covtypes=["categorical"],
        covfits_custom=["NA"],
        time_name="t0",
        obs_data=data,
        return_fits=True,
    )
    covariate_fits, _bounds, _rmses, model_coeffs, model_stderrs, _vcovs, _summary = result
    return covariate_fits["L"], model_coeffs["L"], model_stderrs["L"]


def _later_rows(data: pd.DataFrame) -> pd.DataFrame:
    """Return a copy of the rows with ``t0 > 0``."""
    return data[data["t0"] > 0].copy()


def _drop_late_rows(data: pd.DataFrame, label: int) -> pd.DataFrame:
    """Drop the rows with ``L == label`` and ``t0 > 0``, keeping them at ``t0 == 0``."""
    return data[~((data["L"] == label) & (data["t0"] > 0))]


def _assert_valid_probs(probs: Any, n_rows: int, n_cols: int) -> None:
    """Assert that ``probs`` is a finite probability matrix of the given shape."""
    probs = np.asarray(probs, dtype=float)
    assert probs.shape == (n_rows, n_cols)
    assert np.isfinite(probs).all()
    np.testing.assert_allclose(probs.sum(axis=1), 1.0, atol=1e-8)


@pytest.mark.parametrize("k", [3, 6, 11])
def test_fit_categorical_params_finite(make_cat_data, k):
    """Coefficients and standard errors contain no NaN or inf."""
    _, coeffs, stderrs = _fit(make_cat_data(k))
    assert np.isfinite(np.asarray(coeffs, dtype=float)).all()
    assert np.isfinite(np.asarray(stderrs, dtype=float)).all()


@pytest.mark.parametrize("k", [3, 6, 11])
def test_predict_probs_valid(make_cat_data, k):
    """Predictions at t0 > 0 are finite, have K columns and sum to 1 per row."""
    data = make_cat_data(k)
    fit, _, _ = _fit(data)
    later = _later_rows(data)
    _assert_valid_probs(fit.predict(later), len(later), k)


def test_predict_with_reduced_lag_categories(make_cat_data):
    """Prediction works when lag1_L is a Categorical with fewer levels than in training.

    This mimics what ``update_precoded_history`` does during simulation.
    """
    data = make_cat_data(6)
    fit, _, _ = _fit(data)
    new_df = _later_rows(data)
    new_df = new_df[new_df["lag1_L"].isin([0, 1, 2])].copy()
    new_df["lag1_L"] = pd.Categorical(new_df["lag1_L"])
    assert new_df["lag1_L"].nunique() == 3
    _assert_valid_probs(fit.predict(new_df), len(new_df), 6)


@pytest.mark.parametrize("dropped", [5, 2])
def test_category_only_at_time_zero(make_cat_data, dropped):
    """A category seen only at t0 == 0 does not break the fit or the predictions.

    ``fit_covariate_model`` fits on the rows with t0 > 0, so ``fit.categories``
    holds the sorted labels observed there and the predicted columns are labelled
    with them, whether the dropped label is the last one (5) or a middle one (2).
    """
    data = make_cat_data(6)
    data = data[~((data["L"] == dropped) & (data["t0"] > 0))]
    later = _later_rows(data)
    n_observed = later["L"].nunique()
    assert n_observed == 5
    fit, _, _ = _fit(data)
    _assert_valid_probs(fit.predict(later), len(later), n_observed)
    assert dropped not in list(fit.categories)
    assert len(fit.categories) == n_observed


@pytest.mark.parametrize("dropped", [5, 2])
def test_category_only_at_time_zero_categorical_dtype(make_cat_data, dropped):
    """Same as above, but ``L`` is a pandas Categorical that keeps the unused category.

    The dtype still lists the category absent at t0 > 0. The categories are taken
    from the rows used in the fit, not from the dtype, so the number of predicted
    columns must still equal the number of categories observed at t0 > 0.
    """
    data = make_cat_data(6)
    data["L"] = pd.Categorical(data["L"])
    data = _drop_late_rows(data, dropped)
    later = _later_rows(data)
    n_observed = later["L"].nunique()
    assert n_observed == 5
    fit, _, _ = _fit(data)
    _assert_valid_probs(fit.predict(later), len(later), n_observed)
    assert dropped not in list(fit.categories)
    assert len(fit.categories) == n_observed


@pytest.mark.parametrize("as_categorical", [False, True])
@pytest.mark.parametrize("dropped", [5, 2])
def test_labels_from_full_data_match_predict_columns(make_cat_data, dropped, as_categorical):
    """The stored categories match the predicted columns and the rows used in the fit.

    ``fit_covariate_model`` fits on the rows with t0 > 0, so the category that only
    exists at t0 == 0 must not be in ``categories``. ``simulate`` takes its labels from
    ``categories``, so the labels and the predicted columns cannot shift.
    """
    data = make_cat_data(6)
    if as_categorical:
        data["L"] = pd.Categorical(data["L"])
    data = _drop_late_rows(data, dropped)
    assert (data["L"] == dropped).any()  # the category survives at t0 == 0
    later = _later_rows(data)
    fit, _, _ = _fit(data)
    n_columns = np.asarray(fit.predict(later), dtype=float).shape[1]
    observed = sorted(pd.unique(later["L"].dropna()))
    assert len(fit.categories) == n_columns
    assert list(fit.categories) == observed
    assert dropped not in list(fit.categories)


@pytest.mark.parametrize("k", [3, 6, 11])
def test_multinomial_params_finite(make_cat_data, k):
    """The multinomial wrapper returns finite parameters and standard errors."""
    result = fit_multinomial_formula(FORMULA, make_cat_data(k))
    assert np.isfinite(np.asarray(result.params, dtype=float)).all()
    assert np.isfinite(np.asarray(result.bse, dtype=float)).all()


@pytest.mark.parametrize("k", [3, 6, 11])
def test_multinomial_probs_sum_to_one(make_cat_data, k):
    """Predictions are a finite K-column matrix whose rows sum to 1."""
    data = make_cat_data(k)
    result = fit_multinomial_formula(FORMULA, data)
    probs = result.predict(data)
    _assert_valid_probs(probs, len(data), k)
    assert list(probs.columns) == list(range(k))


def test_multinomial_reference_dropped(make_cat_data):
    """The first category is the reference: K-1 columns, reference first."""
    result = fit_multinomial_formula(FORMULA, make_cat_data(6))
    assert len(result.categories) == 6
    assert result.categories[0] == min(result.categories)
    assert result.params.shape[1] == 5


def test_multinomial_non_contiguous_labels(make_cat_data):
    """Non-contiguous labels are preserved in order in the predicted columns."""
    data = make_cat_data(6)
    data["L"] = (data["L"] + 1) * 10
    data["lag1_L"] = (data["lag1_L"] + 1) * 10
    result = fit_multinomial_formula(FORMULA, data)
    probs = result.predict(data)
    assert list(probs.columns) == [10, 20, 30, 40, 50, 60]


def test_multinomial_with_spline_term(make_cat_data):
    """A spline term and a categorical term fit and predict together."""
    data = make_cat_data(6)
    rng = np.random.default_rng(0)
    data["x"] = rng.uniform(18.0, 89.0, len(data))
    result = fit_multinomial_formula("L ~ bs(x, df=3) + C(lag1_L)", data)
    probs = result.predict(data)
    _assert_valid_probs(probs, len(data), 6)


def test_multinomial_perfectly_collinear_design_raises(make_cat_data):
    """A duplicated predictor column yields a rank error."""
    data = make_cat_data(3)
    data["lag1_L2"] = data["lag1_L"]
    with pytest.raises(ValueError, match="full column rank"):
        fit_multinomial_formula("L ~ lag1_L + lag1_L2", data)


def test_multinomial_probs_with_reduced_lag_levels(make_cat_data):
    """Prediction works when the lag column is categorical with fewer levels."""
    data = make_cat_data(6)
    result = fit_multinomial_formula(FORMULA, data)
    new_df = data[data["lag1_L"].isin([0, 1, 2])].copy()
    new_df["lag1_L"] = pd.Categorical(new_df["lag1_L"])
    assert new_df["lag1_L"].nunique() == 3
    probs = result.predict(new_df)
    _assert_valid_probs(probs, len(new_df), 6)


def test_multinomial_categories_from_rows_used_not_dtype(make_cat_data):
    """Categories come from observed rows, not from an unused dtype category."""
    data = make_cat_data(6)
    data["L"] = pd.Categorical(data["L"], categories=[0, 1, 2, 3, 4, 5, 99])
    result = fit_multinomial_formula(FORMULA, data)
    assert result.categories == [0, 1, 2, 3, 4, 5]
    probs = result.predict(data)
    assert list(probs.columns) == [0, 1, 2, 3, 4, 5]


def test_multinomial_labels_not_shifted_after_gap():
    """Labels are not shifted when a middle category has no rows.

    With labels 0, 1, 3, 4, 5 (2 absent) and a 90% self-transition, predicting
    on ``lag1_L == 4`` must put the highest probability on column 4. A dense
    relabelling would shift it to 3 or 5.
    """
    rng = np.random.default_rng(42)
    n = 4000
    labels = np.array([0, 1, 3, 4, 5])
    lag = rng.choice(labels, size=n)
    outcome = lag.copy()
    mask = rng.random(n) >= 0.9
    outcome[mask] = rng.choice(labels, size=int(mask.sum()))
    data = pd.DataFrame({"L": outcome, "lag1_L": lag})
    result = fit_multinomial_formula("L ~ C(lag1_L)", data)
    assert result.categories == [0, 1, 3, 4, 5]
    new_df = pd.DataFrame({"lag1_L": [4] * 200})
    probs = result.predict(new_df)
    assert probs.mean(axis=0).idxmax() == 4


def test_multinomial_unused_predictor_levels(make_cat_data):
    """An unused predictor level does not break the rank check or prediction."""
    data = make_cat_data(6)
    data["lag1_L"] = pd.Categorical(data["lag1_L"], categories=[0, 1, 2, 3, 4, 5, 99])
    result = fit_multinomial_formula(FORMULA, data)
    probs = result.predict(data)
    _assert_valid_probs(probs, len(data), 6)


def test_multinomial_nan_newton_falls_back_to_bfgs(make_cat_data, monkeypatch):
    """A Newton fit with NaN parameters and ``converged=True`` falls back to BFGS.

    statsmodels' Newton optimiser can report convergence while every parameter
    is NaN. The wrapper must reject such a fit on finiteness, not on the
    ``converged`` flag alone, and move on to the next method.
    """
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
        result = fit_multinomial_formula(FORMULA, make_cat_data(6))

    assert result.fit_method == "bfgs"
    assert np.isfinite(np.asarray(result.params, dtype=float)).all()
    assert result.attempts[0]["converged"] is True
    assert result.attempts[0]["finite"] is False


def test_multinomial_converged_false_is_rejected(make_cat_data, monkeypatch):
    """A finite fit that reports ``converged=False`` is rejected in favour of BFGS."""
    real_fit = MNLogit.fit

    def fake_fit(self, *args, **kwargs):
        if kwargs.get("method") == "newton":
            return types.SimpleNamespace(
                params=np.zeros((1, 1)),
                llf=0.0,
                bse=np.zeros((1, 1)),
                mle_retvals={"converged": False},
            )
        return real_fit(self, *args, **kwargs)

    monkeypatch.setattr(MNLogit, "fit", fake_fit)
    with pytest.warns(UserWarning, match="bfgs"):
        result = fit_multinomial_formula(FORMULA, make_cat_data(6))

    assert result.fit_method == "bfgs"
    assert result.attempts[0]["converged"] is False
    assert result.attempts[0]["finite"] is True


def test_multinomial_all_attempts_fail_raises(make_cat_data, monkeypatch):
    """When every attempt, including the regularized one, returns NaN, an error is raised."""

    def nan_result(self, *args, **kwargs):
        return types.SimpleNamespace(
            params=np.full((1, 1), np.nan),
            llf=float("nan"),
            bse=np.full((1, 1), np.nan),
            mle_retvals={"converged": True},
        )

    monkeypatch.setattr(MNLogit, "fit", nan_result)
    monkeypatch.setattr(MNLogit, "fit_regularized", nan_result)
    with pytest.raises(ValueError, match=r"covariate 'L'.*merging rare categories"):
        fit_multinomial_formula(FORMULA, make_cat_data(6))


def test_stable_softmax_large_values(make_cat_data):
    """The stable softmax does not overflow on large logits, and predict stays finite."""
    from pygformula.utils.multinomial import _stable_softmax

    for row in (
        np.array([1000.0, 0.0, -1000.0]),
        np.array([-1000.0, -1000.0, -1000.0]),
    ):
        probs = _stable_softmax(row)
        assert np.isfinite(probs).all()
        np.testing.assert_allclose(probs.sum(), 1.0, atol=1e-12)

    data = make_cat_data(6)
    result = fit_multinomial_formula(FORMULA, data)
    scaled = np.asarray(result.params, dtype=float) * 1e3
    result._result = types.SimpleNamespace(params=scaled)
    probs = result.predict(data)
    _assert_valid_probs(probs, len(data), 6)


def test_multinomial_quasi_separation_contract():
    """Quasi-separated data still yields finite parameters and valid probabilities.

    The data has an empty ``(lag1_L, L)`` cell, near-deterministic transitions
    and an unscaled continuous predictor in ``[18, 89]``. The regularized
    fallback guarantees a finite fit even if Newton diverges.
    """
    rng = np.random.default_rng(7)
    n = 4000
    lag = rng.integers(0, 6, size=n)
    outcome = lag.copy()
    flip = rng.random(n) < 0.03
    outcome[flip] = (lag[flip] + 1) % 6
    x = rng.uniform(18.0, 89.0, size=n)
    data = pd.DataFrame({"L": outcome, "lag1_L": lag, "x": x})

    result = fit_multinomial_formula("L ~ C(lag1_L) + x", data)
    assert np.isfinite(np.asarray(result.params, dtype=float)).all()
    probs = result.predict(data)
    _assert_valid_probs(probs, n, 6)


def test_multinomial_l1_result_without_covariance(make_cat_data, monkeypatch):
    """The l1 fallback result has no covariance: bse and cov_params give NaN, not an error."""

    class _NoCovariance:
        # 7 = len(spec.column_names) for FORMULA (Intercept + 5 C(lag1_L) dummies + t0).
        params = np.ones((7, 5))
        llf = -1.0
        mle_retvals = {"converged": True}

        @property
        def bse(self):
            raise ValueError("need covariance of parameters")

        def cov_params(self):
            raise ValueError("need covariance of parameters")

    def nan_result(self, *args, **kwargs):
        return types.SimpleNamespace(
            params=np.full((1, 1), np.nan),
            llf=float("nan"),
            bse=np.full((1, 1), np.nan),
            mle_retvals={"converged": True},
        )

    monkeypatch.setattr(MNLogit, "fit", nan_result)
    monkeypatch.setattr(MNLogit, "fit_regularized", lambda self, *a, **k: _NoCovariance())
    with pytest.warns(UserWarning):
        result = fit_multinomial_formula(FORMULA, make_cat_data(6))

    assert result.fit_method == "l1"
    assert np.isnan(np.asarray(result.bse, dtype=float)).all()
    assert result.bse.shape == (7, 5)
    assert np.isnan(result.cov_params()).all()
    assert result.cov_params().shape == (35, 35)


def test_fit_covariate_model_categorical_returns_single_model(make_cat_data):
    """A categorical covariate yields one fit entry with K - 1 parameter columns."""
    from pygformula.utils.formula import apply_design

    data = make_cat_data(11)
    result = fit_covariate_model(
        covmodels=[FORMULA],
        covnames=["L"],
        covtypes=["categorical"],
        covfits_custom=["NA"],
        time_name="t0",
        obs_data=data,
        return_fits=True,
    )
    covariate_fits, _, _, model_coeffs, _, _, _ = result
    assert list(covariate_fits.keys()) == ["L"]
    fit = covariate_fits["L"]
    n_predictors = apply_design(fit.spec, data.iloc[:5]).shape[1]
    assert fit.params.shape == (n_predictors, 10)
    assert np.asarray(model_coeffs["L"]).shape == (n_predictors, 10)


def test_fit_covariate_model_categorical_survives_nan_newton(make_cat_data, monkeypatch):
    """With Newton returning NaN and converged=True, the covariate fit is still finite."""
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
        fit, coeffs, _ = _fit(make_cat_data(6))
    assert fit.fit_method == "bfgs"
    assert np.isfinite(np.asarray(coeffs, dtype=float)).all()


def test_multinomial_summary_without_covariance(make_cat_data, monkeypatch):
    """summary() returns a plain string when the accepted fit has no covariance.

    The mock makes Newton/BFGS/L-BFGS return NaN and the l1 fallback return a
    result without a covariance matrix. summary() must not raise and must
    return a ``str``, while a normal fit still returns the statsmodels object.
    """

    class _NoCovariance:
        # 7 predictors (Intercept + 5 C(lag1_L) dummies + t0) x 5 columns.
        params = np.ones((7, 5))
        llf = -1.0
        mle_retvals = {"converged": True}

        @property
        def bse(self):
            raise ValueError("need covariance of parameters")

        def cov_params(self):
            raise ValueError("need covariance of parameters")

    def nan_result(self, *args, **kwargs):
        return types.SimpleNamespace(
            params=np.full((1, 1), np.nan),
            llf=float("nan"),
            bse=np.full((1, 1), np.nan),
            mle_retvals={"converged": True},
        )

    normal = fit_multinomial_formula(FORMULA, make_cat_data(6))
    assert not isinstance(normal.summary(), str)

    monkeypatch.setattr(MNLogit, "fit", nan_result)
    monkeypatch.setattr(MNLogit, "fit_regularized", lambda self, *a, **k: _NoCovariance())
    with pytest.warns(UserWarning):
        result = fit_multinomial_formula(FORMULA, make_cat_data(6))

    summary = result.summary()
    assert isinstance(summary, str)
    assert "covariate 'L'" in summary
    assert "not available" in summary


def test_multinomial_params_are_labelled(make_cat_data):
    """params and bse are DataFrames labelled by design columns and categories."""
    fit = fit_multinomial_formula(FORMULA, make_cat_data(6))
    n_predictors = len(list(fit.spec.column_names))

    assert isinstance(fit.params, pd.DataFrame)
    assert list(fit.params.columns) == fit.categories[1:]
    assert list(fit.params.index) == list(fit.spec.column_names)
    assert fit.params.shape == (n_predictors, 5)
    assert np.isfinite(fit.params.to_numpy()).all()

    assert isinstance(fit.bse, pd.DataFrame)
    assert list(fit.bse.columns) == fit.categories[1:]
    assert list(fit.bse.index) == list(fit.spec.column_names)
    assert fit.bse.shape == (n_predictors, 5)
    assert np.isfinite(fit.bse.to_numpy()).all()


def test_multinomial_predict_with_duplicate_index(make_cat_data):
    """predict keeps the length and the (duplicated) index of new_data."""
    data = make_cat_data(6)
    result = fit_multinomial_formula(FORMULA, data)
    new_df = data.iloc[:40].copy()
    new_df.index = pd.Index([i // 2 for i in range(len(new_df))])
    assert not new_df.index.is_unique
    probs = result.predict(new_df)
    _assert_valid_probs(probs, len(new_df), 6)
    assert list(probs.index) == list(new_df.index)
    expected = result.predict(new_df.reset_index(drop=True)).to_numpy()
    np.testing.assert_allclose(probs.to_numpy(), expected)


def test_multinomial_real_l1_fallback(make_cat_data, monkeypatch):
    """With Newton, BFGS and L-BFGS rejected, the real regularized fit is accepted."""
    real_fit = MNLogit.fit

    def fake_fit(self, *args, **kwargs):
        if kwargs.get("method") in ("newton", "bfgs", "lbfgs"):
            return types.SimpleNamespace(
                params=np.full((1, 1), np.nan),
                llf=float("nan"),
                bse=np.full((1, 1), np.nan),
                mle_retvals={"converged": True},
            )
        return real_fit(self, *args, **kwargs)

    monkeypatch.setattr(MNLogit, "fit", fake_fit)
    data = make_cat_data(6)
    with pytest.warns(UserWarning, match="l1"):
        result = fit_multinomial_formula(FORMULA, data)

    assert result.fit_method == "l1"
    assert np.isfinite(np.asarray(result.params, dtype=float)).all()
    _assert_valid_probs(result.predict(data), len(data), 6)



def test_multinomial_rank_check_ignores_predictor_scale(make_cat_data):
    """Powers of an unscaled predictor are not reported as collinear."""
    data = make_cat_data(3)
    rng = np.random.default_rng(0)
    data["year"] = rng.integers(1968, 2020, len(data)).astype(float)
    try:
        fit_multinomial_formula("L ~ year + I(year**2) + I(year**3)", data)
    except ValueError as exc:
        assert "full column rank" not in str(exc)
