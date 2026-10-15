import math
import warnings
from typing import Any, List

import numpy as np
import pandas as pd
import pytest
import statsmodels.api as sm
import statsmodels.formula.api as smf

from pygformula.parametric_gformula.fit import (
    fit_censor_model,
    fit_compevent_model,
    fit_covariate_model,
    fit_ymodel,
)


@pytest.fixture
def cov_data() -> pd.DataFrame:
    """Simulated person-time data with a lagged covariate and a binary treatment."""
    rng = np.random.default_rng(42)
    n = 100
    return pd.DataFrame(
        {
            "id": np.arange(n),
            "t0": rng.integers(0, 5, size=n),
            "L2": rng.normal(0, 1, size=n),
            "lag1_L2": rng.normal(0, 1, size=n),
            "A": rng.integers(0, 2, size=n),
        }
    )


def _fit_l2(covmodels: List[str], data: pd.DataFrame) -> Any:
    """Fit the normal covariate model for L2 and return its fit object."""
    covariate_fits, *_ = fit_covariate_model(
        covmodels=covmodels,
        covnames=["L2"],
        covtypes=["normal"],
        covfits_custom=None,
        time_name="t0",
        obs_data=data,
        return_fits=True,
    )
    # The container of fits may be a dict keyed by covariate name or a list.
    return covariate_fits["L2"] if isinstance(covariate_fits, dict) else covariate_fits[0]


# --------------------------------------------------------------------------- #
# Formula parsing
# --------------------------------------------------------------------------- #
def test_fit_covariate_model_with_spline_term(cov_data: pd.DataFrame) -> None:
    fit = _fit_l2(["L2 ~ bs(lag1_L2, df=4) + C(A) + t0"], cov_data)

    params = np.asarray(fit.params, dtype=float)
    assert params.size > 0
    assert np.all(np.isfinite(params))


def test_fit_covariate_model_double_space_formula(cov_data: pd.DataFrame) -> None:
    fit_spaced = _fit_l2(["L2 ~  lag1_L2  +  t0"], cov_data)
    fit_plain = _fit_l2(["L2 ~ lag1_L2 + t0"], cov_data)

    assert np.all(np.isfinite(np.asarray(fit_spaced.params, dtype=float)))
    pd.testing.assert_series_equal(fit_spaced.params, fit_plain.params, atol=1e-10)


# --------------------------------------------------------------------------- #
# fit_glm_formula in fit_covariate_model
# --------------------------------------------------------------------------- #
def test_binary_cov_fit_matches_previous(cov_data: pd.DataFrame) -> None:
    covariate_fits, _, _, _, _, _, _ = fit_covariate_model(
        covmodels=["A ~ L2"],
        covnames=["A"],
        covtypes=["binary"],
        covfits_custom=None,
        time_name="t0",
        obs_data=cov_data,
        return_fits=True,
    )
    fit = covariate_fits["A"]

    # fit_covariate_model only uses rows with t0 > 0, so the oracle must do the same.
    oracle_data = cov_data[cov_data["t0"] > 0]
    oracle_fit = smf.glm("A ~ L2", data=oracle_data, family=sm.families.Binomial()).fit()

    # Compare by label, never by position.
    pd.testing.assert_series_equal(
        fit.params.sort_index(),
        oracle_fit.params.sort_index(),
        atol=1e-6,
        check_names=False,
    )


def test_normal_and_bounded_rmse_unchanged(cov_data: pd.DataFrame) -> None:
    oracle_data = cov_data[cov_data["t0"] > 0]

    # --- Normal ---
    _, _, rmses_norm, _, _, _, _ = fit_covariate_model(
        covmodels=["L2 ~ lag1_L2"],
        covnames=["L2"],
        covtypes=["normal"],
        covfits_custom=None,
        time_name="t0",
        obs_data=cov_data,
        return_fits=True,
    )

    oracle_fit_norm = smf.glm("L2 ~ lag1_L2", data=oracle_data, family=sm.families.Gaussian()).fit()
    oracle_rmse_norm = np.sqrt(np.mean((oracle_fit_norm.predict() - oracle_data["L2"]) ** 2))

    assert np.isclose(rmses_norm["L2"], oracle_rmse_norm, atol=1e-6)

    # --- Bounded normal (L2 is rescaled to [0, 1] with the observed min and max) ---
    _, _, rmses_bound, _, _, _, _ = fit_covariate_model(
        covmodels=["L2 ~ lag1_L2"],
        covnames=["L2"],
        covtypes=["bounded normal"],
        covfits_custom=None,
        time_name="t0",
        obs_data=cov_data,
        return_fits=True,
    )

    min_cov = cov_data["L2"].min()
    max_cov = cov_data["L2"].max()
    bound_data = oracle_data.copy()
    bound_data["L2"] = (bound_data["L2"] - min_cov) / (max_cov - min_cov)

    oracle_fit_bound = smf.glm("L2 ~ lag1_L2", data=bound_data, family=sm.families.Gaussian()).fit()
    oracle_rmse_bound = np.sqrt(np.mean((oracle_fit_bound.predict() - bound_data["L2"]) ** 2))

    assert np.isclose(rmses_bound["L2"], oracle_rmse_bound, atol=1e-6)


def test_zero_inflated_two_fits() -> None:
    # L2 has exact zeros and strictly positive values: the non-zero part is log-transformed.
    rng = np.random.default_rng(42)
    n = 200
    is_zero = rng.random(n) < 0.4
    data = pd.DataFrame(
        {
            "id": np.arange(n),
            "t0": rng.integers(0, 5, size=n),
            "L2": np.where(is_zero, 0.0, rng.lognormal(0.0, 0.5, size=n)),
            "lag1_L2": rng.normal(0, 1, size=n),
        }
    )

    covariate_fits, _, _, _, _, _, _ = fit_covariate_model(
        covmodels=["L2 ~ lag1_L2"],
        covnames=["L2"],
        covtypes=["zero-inflated normal"],
        covfits_custom=None,
        time_name="t0",
        obs_data=data,
        return_fits=True,
    )

    assert isinstance(covariate_fits["L2"], list)
    assert len(covariate_fits["L2"]) == 2

    indicator_fit, non_zero_fit = covariate_fits["L2"]

    new_data = pd.DataFrame(
        {
            "L2": [1.0, 2.0, 0.0],
            "lag1_L2": [1.1, 2.1, 0.1],
            "t0": [1, 2, 3],
            "id": [100, 101, 102],
        }
    )
    new_data["I_L2"] = (new_data["L2"] != 0).astype(int)
    new_data["log_L2"] = new_data["L2"].apply(lambda x: math.log(x) if x != 0 else 0.0)

    pred_indicator = np.asarray(indicator_fit.predict(new_data), dtype=float)
    assert np.all(np.isfinite(pred_indicator))

    pred_nonzero = np.asarray(non_zero_fit.predict(new_data[new_data["L2"] != 0]), dtype=float)
    assert np.all(np.isfinite(pred_nonzero))


@pytest.mark.filterwarnings("ignore:The design matrix is rank-deficient")
def test_absorbing_fit_runs() -> None:
    rng = np.random.default_rng(42)
    n = 200
    data = pd.DataFrame(
        {
            "id": np.arange(n),
            "t0": rng.integers(0, 5, size=n),
            "A": rng.integers(0, 2, size=n),
            "lag1_A": rng.integers(0, 2, size=n),
            "L2": rng.normal(0, 1, size=n),
        }
    )

    # fit_covariate_model keeps only the formula variables and then filters on the time
    # column (t0 > 0) and on lag1_A == 0, so both t0 and lag1_A must appear in the formula.
    # Inside the absorbing subset lag1_A is constant (all zeros, so the design is rank-deficient);
    # the GLM still fits (pseudo-inverse) and the parameters must be finite.
    covariate_fits, _, _, _, _, _, _ = fit_covariate_model(
        covmodels=["A ~ lag1_A + L2 + t0"],
        covnames=["A"],
        covtypes=["absorbing"],
        covfits_custom=None,
        time_name="t0",
        obs_data=data,
        return_fits=True,
    )

    assert "A" in covariate_fits
    fit = covariate_fits["A"]
    assert hasattr(fit, "params") and hasattr(fit, "predict")
    assert np.all(np.isfinite(np.asarray(fit.params, dtype=float)))


def test_fit_ymodel_with_spline_survival() -> None:
    rng = np.random.default_rng(42)
    n = 100
    data = pd.DataFrame(
        {
            "t0": rng.integers(0, 5, size=n),
            "L2": rng.normal(0, 1, size=n),
            "A": rng.integers(0, 2, size=n),
            "Y": rng.binomial(1, 0.5, size=n),
        }
    )
    ymodel = "Y ~ bs(L2, df=4) + A + t0"

    fit, *_ = fit_ymodel(
        ymodel=ymodel,
        outcome_type="survival",
        outcome_name="Y",
        ymodel_fit_custom=None,
        time_name="t0",
        obs_data=data,
        competing=False,
        compevent_name=None,
        return_fits=True,
        yrestrictions=None,
    )
    assert np.all(np.isfinite(np.asarray(fit.params, dtype=float)))

    # New data inside the training range of L2 (a bs term cannot extrapolate).
    new_data = pd.DataFrame(
        {
            "L2": np.quantile(data["L2"], [0.1, 0.5, 0.9]),
            "A": [0, 1, 0],
            "t0": [1, 2, 3],
        }
    )
    predictions = np.asarray(fit.predict(new_data), dtype=float)
    assert predictions.shape == (3,)
    assert np.all(np.isfinite(predictions))

    # Spline column names differ between libraries, so compare predictions, not labels.
    oracle_fit = smf.glm(ymodel, data=data, family=sm.families.Binomial()).fit()
    np.testing.assert_allclose(
        predictions, oracle_fit.predict(new_data).to_numpy(), rtol=0, atol=1e-6
    )


def test_fit_ymodel_matches_smf_plain() -> None:
    rng = np.random.default_rng(42)
    n = 100
    data = pd.DataFrame(
        {
            "t0": rng.integers(0, 5, size=n),
            "A": rng.integers(0, 2, size=n),
            "Y": rng.normal(0, 1, size=n),
        }
    )
    ymodel = "Y ~ A + t0"

    fit, *_ = fit_ymodel(
        ymodel=ymodel,
        outcome_type="continuous_eof",
        outcome_name="Y",
        ymodel_fit_custom=None,
        time_name="t0",
        obs_data=data,
        competing=False,
        compevent_name=None,
        return_fits=True,
        yrestrictions=None,
    )

    # fit_ymodel uses the rows with t0 >= 0 and a non-missing outcome: all rows here.
    oracle_fit = smf.glm(ymodel, data=data, family=sm.families.Gaussian()).fit()
    pd.testing.assert_series_equal(
        fit.params.sort_index(),
        oracle_fit.params.sort_index(),
        atol=1e-6,
        check_names=False,
    )


def test_fit_compevent_model_runs() -> None:
    rng = np.random.default_rng(42)
    n = 100
    data = pd.DataFrame(
        {
            "t0": rng.integers(0, 5, size=n),
            "A": rng.integers(0, 2, size=n),
            "C": rng.binomial(1, 0.5, size=n),
        }
    )

    fit, *_ = fit_compevent_model(
        compevent_model="C ~ A + t0",
        compevent_name="C",
        time_name="t0",
        obs_data=data,
        return_fits=True,
        compevent_restrictions=None,
    )

    assert set(fit.params.index) == {"Intercept", "A", "t0"}
    assert np.all(np.isfinite(np.asarray(fit.params, dtype=float)))


def test_fit_censor_model_with_column_named_C() -> None:
    # The censoring column is called "C", the same name as formulaic's C() transform.
    rng = np.random.default_rng(42)
    n = 100
    data = pd.DataFrame(
        {
            "t0": rng.integers(0, 5, size=n),
            "A": rng.normal(0, 1, size=n),
            "L": rng.normal(0, 1, size=n),
            "C": rng.binomial(1, 0.5, size=n),
        }
    )

    fit, *_ = fit_censor_model(
        censor_model="C ~ A + L",
        censor_name="C",
        time_name="t0",
        obs_data=data,
        return_fits=True,
    )

    # C is the response, not a categorical transform: no C-related parameter appears.
    assert set(fit.params.index) == {"Intercept", "A", "L"}
    assert np.all(np.isfinite(np.asarray(fit.params, dtype=float)))

    new_data = pd.DataFrame({"A": [0.0, 1.0], "L": [-0.5, 0.5]})
    assert np.all(np.isfinite(np.asarray(fit.predict(new_data), dtype=float)))


def test_truncated_normal_rejects_spline_formula(cov_data: pd.DataFrame) -> None:
    """A truncated normal covariate with a spline term raises a clear ValueError.

    Args:
        cov_data: Fixture with the simulated covariate data.
    """
    with pytest.raises(ValueError, match="only plain terms"):
        fit_covariate_model(
            covmodels=["L2 ~ bs(lag1_L2, df=4) + t0"],
            covnames=["L2"],
            covtypes=["truncated normal"],
            covfits_custom=["NA"],
            time_name="t0",
            obs_data=cov_data,
            return_fits=True,
            trunc_params=[[1, "right"]],
        )


def test_fit_ymodel_drops_rows_with_missing_predictors() -> None:
    rng = np.random.default_rng(42)
    n = 100
    data = pd.DataFrame(
        {
            "t0": rng.integers(0, 5, size=n),
            "L2": rng.normal(0, 1, size=n),
            "A": rng.integers(0, 2, size=n),
            "Y": rng.binomial(1, 0.5, size=n),
        }
    )
    data.loc[:9, "L2"] = np.nan
    ymodel = "Y ~ L2 + A + t0"

    with pytest.warns(UserWarning, match="dropped"):
        fit, *_ = fit_ymodel(
            ymodel=ymodel,
            outcome_type="survival",
            outcome_name="Y",
            ymodel_fit_custom=None,
            time_name="t0",
            obs_data=data,
            competing=False,
            compevent_name=None,
            return_fits=True,
            yrestrictions=None,
        )

    assert np.all(np.isfinite(np.asarray(fit.params, dtype=float)))


def test_fit_ymodel_custom_fit_receives_rows_with_missing() -> None:
    """A custom outcome fit must receive all rows, including rows with NaN predictors."""
    rng = np.random.default_rng(42)
    n = 100
    data = pd.DataFrame(
        {
            "t0": rng.integers(0, 5, size=n),
            "L2": rng.normal(0, 1, size=n),
            "A": rng.integers(0, 2, size=n),
            "Y": rng.binomial(1, 0.5, size=n),
        }
    )
    data.loc[:9, "L2"] = np.nan
    ymodel = "Y ~ L2 + A + t0"

    seen: List[int] = []

    def custom(ymodel: str, data: pd.DataFrame) -> object:
        seen.append(len(data))
        return object()

    with warnings.catch_warnings():
        warnings.filterwarnings("error", message=".*have missing values.*")
        fit_ymodel(
            ymodel=ymodel,
            outcome_type="survival",
            outcome_name="Y",
            ymodel_fit_custom=custom,
            time_name="t0",
            obs_data=data,
            competing=False,
            compevent_name=None,
            return_fits=False,
            yrestrictions=None,
        )

    assert seen == [n]


def test_fit_covariate_model_with_column_named_C_as_predictor(cov_data: pd.DataFrame) -> None:
    """A predictor column named C is kept when fit_covariate_model selects columns."""
    data = cov_data.copy()
    data["C"] = np.random.default_rng(1).normal(size=len(data))
    fit = _fit_l2(["L2 ~ C + t0"], data)

    assert set(fit.params.index) == {"Intercept", "C", "t0"}
    assert np.all(np.isfinite(np.asarray(fit.params, dtype=float)))
