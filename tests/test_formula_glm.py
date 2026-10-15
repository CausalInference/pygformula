import warnings

import numpy as np
import pandas as pd
import pytest
import statsmodels.api as sm
import statsmodels.formula.api as smf
from statsmodels.tools.sm_exceptions import SingularMatrixWarning

import pygformula.utils.formula as formula_module
from pygformula.utils.formula import fit_glm_formula


def _assert_series_match(ours: pd.Series, ref: pd.Series) -> None:
    """Compare two Series by label, ignoring the order of the labels."""
    assert set(ours.index) == set(ref.index)
    pd.testing.assert_series_equal(
        ours.reindex(ref.index),
        ref,
        check_names=False,
        check_index_type=False,
        atol=1e-6,
        rtol=0,
    )


def test_glm_binomial_matches_smf() -> None:
    rng = np.random.default_rng(42)
    data = pd.DataFrame(
        {
            "y": rng.binomial(1, 0.5, 100),
            "x1": rng.standard_normal(100),
            "g": rng.choice(["A", "B"], 100),
        }
    )
    formula = "y ~ x1 + C(g)"
    ours = fit_glm_formula(formula, data, family=sm.families.Binomial())
    ref = smf.glm(formula, data, family=sm.families.Binomial()).fit()

    _assert_series_match(ours.params, ref.params)
    _assert_series_match(ours.bse, ref.bse)


def test_glm_gaussian_matches_smf() -> None:
    rng = np.random.default_rng(42)
    data = pd.DataFrame(
        {
            "y": rng.standard_normal(100),
            "x1": rng.standard_normal(100),
            "x2": rng.standard_normal(100),
        }
    )
    formula = "y ~ x1 + x2"
    ours = fit_glm_formula(formula, data, family=sm.families.Gaussian())
    ref = smf.glm(formula, data, family=sm.families.Gaussian()).fit()

    _assert_series_match(ours.params, ref.params)
    _assert_series_match(ours.bse, ref.bse)


def test_glm_predict_new_data_index() -> None:
    rng = np.random.default_rng(42)
    data = pd.DataFrame({"y": rng.standard_normal(100), "x1": rng.standard_normal(100)})
    res = fit_glm_formula("y ~ x1", data, family=sm.families.Gaussian())

    new_data = pd.DataFrame({"x1": rng.standard_normal(10)}, index=np.arange(100, 110))
    predictions = res.predict(new_data)

    assert predictions.index.equals(new_data.index)
    assert len(predictions) == 10


def test_glm_predict_none_returns_fitted() -> None:
    rng = np.random.default_rng(42)
    data = pd.DataFrame({"y": rng.standard_normal(100), "x1": rng.standard_normal(100)})
    res = fit_glm_formula("y ~ x1", data, family=sm.families.Gaussian())

    predictions = res.predict(None)

    assert predictions.name == "fitted"
    assert len(predictions) == 100
    np.testing.assert_allclose(predictions.to_numpy(), res.fittedvalues.to_numpy())


def test_glm_spline_predict_matches_smf() -> None:
    rng = np.random.default_rng(42)
    data = pd.DataFrame(
        {
            "y": rng.standard_normal(100),
            "x": np.linspace(0, 10, 100),
            "z": rng.standard_normal(100),
        }
    )
    formula = "y ~ bs(x, df=4) + z"
    ours = fit_glm_formula(formula, data, family=sm.families.Gaussian())
    ref = smf.glm(formula, data, family=sm.families.Gaussian()).fit()

    # Same number of parameters; names of spline columns may differ by library.
    assert len(ours.params) == len(ref.params)

    new_data = pd.DataFrame({"x": [2.5, 5.0, 7.5], "z": [1.0, -0.5, 0.3]})
    pd.testing.assert_series_equal(
        ours.predict(new_data),
        ref.predict(new_data),
        check_names=False,
        atol=1e-6,
        rtol=0,
    )


def test_glm_nonfinite_fit_raises(monkeypatch: pytest.MonkeyPatch) -> None:
    class _FakeResult:
        params = pd.Series([np.nan, 1.0])
        llf = -1.0

    class _FakeGLM:
        def __init__(self, *args: object, **kwargs: object) -> None:
            pass

        def fit(self) -> _FakeResult:
            return _FakeResult()

    monkeypatch.setattr(formula_module.sm, "GLM", _FakeGLM)
    rng = np.random.default_rng(1)
    data = pd.DataFrame({"y": rng.standard_normal(20), "x1": rng.standard_normal(20)})

    with pytest.raises(ValueError, match="non-finite"):
        fit_glm_formula("y ~ x1", data, family=sm.families.Gaussian())


def test_glm_unused_categorical_level_not_rank_deficient() -> None:
    rng = np.random.default_rng(7)
    data = pd.DataFrame(
        {
            "y": rng.binomial(1, 0.5, 200),
            "g": pd.Categorical(
                rng.choice(["a", "b"], 200), categories=["a", "b", "z"]
            ),
            "x": rng.standard_normal(200),
        }
    )

    with warnings.catch_warnings():
        warnings.simplefilter("error", SingularMatrixWarning)
        res = fit_glm_formula("y ~ C(g) + x", data, sm.families.Binomial())

    assert not any("[T.z]" in name for name in res.params.index)
    assert np.all(np.isfinite(res.params.to_numpy()))

    pred = res.predict(data[data["g"].isin(["a", "b"])])
    assert np.all(np.isfinite(pred.to_numpy()))

    bad_row = pd.DataFrame(
        {"g": pd.Categorical(["z"], categories=["a", "b", "z"]), "x": [0.0]}
    )
    with pytest.raises(ValueError):
        res.predict(bad_row)

    assert "z" in data["g"].cat.categories


def test_drop_rows_with_missing_warns_and_drops() -> None:
    rng = np.random.default_rng(42)
    data = pd.DataFrame(
        {
            "y": rng.standard_normal(200),
            "x1": rng.standard_normal(200),
            "x2": rng.standard_normal(200),
        }
    )
    data.loc[:9, "x1"] = np.nan

    with pytest.warns(UserWarning, match="dropped"):
        result = formula_module.drop_rows_with_missing("y ~ x1 + x2", data, "ymodel")

    assert len(result) == 190
    assert not result[["y", "x1", "x2"]].isna().any().any()
    assert len(data) == 200


def test_drop_rows_with_missing_without_nan_is_silent() -> None:
    rng = np.random.default_rng(42)
    data = pd.DataFrame(
        {
            "y": rng.standard_normal(50),
            "x1": rng.standard_normal(50),
        }
    )

    with warnings.catch_warnings():
        warnings.simplefilter("error")
        result = formula_module.drop_rows_with_missing("y ~ x1", data, "ymodel")

    assert result is data
