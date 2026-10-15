"""Tests for the formulaic-based design-matrix layer in `pygformula.utils.formula`.

The oracle for the plain and categorical cases is statsmodels' formula API
(patsy), used here only as a reference, never inside the package itself.
"""
import numpy as np
import pandas as pd
import pytest
import statsmodels.api as sm
import statsmodels.formula.api as smf

from pygformula.utils.formula import (
    _variable_name,
    apply_design,
    build_design,
    fit_glm_formula,
    get_formula_variables,
)


@pytest.fixture
def plain_data() -> pd.DataFrame:
    """Simulate a small DataFrame with a binary outcome and two covariates.

    Returns:
        A DataFrame with columns `y`, `x1`, `x2` and `t0`.
    """
    rng = np.random.default_rng(0)
    n = 60
    return pd.DataFrame(
        {
            "y": rng.binomial(1, 0.4, n),
            "x1": rng.normal(size=n),
            "x2": rng.normal(size=n),
            "t0": rng.integers(0, 3, n),
        }
    )


def test_plain_formula_matches_patsy(plain_data: pd.DataFrame) -> None:
    """`build_design` must reproduce patsy's design matrix for a plain formula."""
    formula = "y ~ x1 + x2 + t0"
    y, X, spec = build_design(formula, plain_data)

    reference = smf.glm(formula, plain_data, family=sm.families.Binomial()).fit()

    assert list(X.columns) == list(reference.model.exog_names)
    np.testing.assert_allclose(X.to_numpy(dtype=float), reference.model.exog)
    assert y is not None
    np.testing.assert_allclose(y.to_numpy(dtype=float), plain_data["y"].to_numpy(dtype=float))


def test_categorical_treatment_coding() -> None:
    """A `C(g)` term must produce K-1 treatment-coded dummy columns."""
    rng = np.random.default_rng(1)
    n = 60
    data = pd.DataFrame(
        {
            "y": rng.binomial(1, 0.4, n),
            "g": rng.choice(["a", "b", "c"], size=n),
            "x": rng.normal(size=n),
        }
    )
    y, X, spec = build_design("y ~ C(g) + x", data)

    assert list(X.columns) == ["Intercept", "C(g)[T.b]", "C(g)[T.c]", "x"]
    assert y is not None


def test_formula_syntax_supported() -> None:
    """Interaction, polynomial, log and no-intercept syntaxes must all build."""
    rng = np.random.default_rng(2)
    n = 60
    data = pd.DataFrame(
        {
            "y": rng.binomial(1, 0.4, n),
            "x1": rng.uniform(1, 5, n),
            "x2": rng.normal(size=n),
        }
    )

    cases = [
        ("y ~ x1*x2", 4),
        ("y ~ x1:x2 + x1", 3),
        ("y ~ I(x1**2) + x2", 3),
        ("y ~ log(x1) + x2", 3),
        ("y ~ x1 + x2 - 1", 2),
    ]
    for formula, n_cols in cases:
        y, X, spec = build_design(formula, data)
        assert X.shape == (n, n_cols), formula
        assert y is not None

        variables = get_formula_variables(formula)
        assert variables == sorted(set(variables))
        for variable in variables:
            assert variable in data.columns, (formula, variable)


def test_column_named_C_is_data() -> None:
    """A data column literally named `C` must not be confused with `C()`."""
    rng = np.random.default_rng(3)
    n = 60
    data = pd.DataFrame(
        {
            "C": rng.binomial(1, 0.3, n),
            "A": rng.normal(size=n),
            "L": rng.normal(size=n),
        }
    )
    y, X, spec = build_design("C ~ A + L", data)

    assert y is not None
    assert list(X.columns) == ["Intercept", "A", "L"]


def test_apply_design_keeps_levels() -> None:
    """Applying a stored spec to data with a subset of levels keeps all columns."""
    rng = np.random.default_rng(4)
    n = 60
    train = pd.DataFrame(
        {
            "y": rng.binomial(1, 0.4, n),
            "g": rng.choice(["a", "b", "c"], size=n),
            "x": rng.normal(size=n),
        }
    )
    _, X_train, spec = build_design("y ~ C(g) + x", train)

    new_data = pd.DataFrame({"g": ["c", "c", "c"], "x": [0.1, 0.2, 0.3]})
    X_new = apply_design(spec, new_data)

    assert list(X_new.columns) == list(X_train.columns)
    assert len(X_new) == len(new_data)


def test_apply_design_unseen_level_raises() -> None:
    """An unseen categorical level in new data must raise a clear ValueError."""
    rng = np.random.default_rng(5)
    n = 60
    train = pd.DataFrame(
        {
            "y": rng.binomial(1, 0.4, n),
            "g": rng.choice(["a", "b", "c"], size=n),
            "x": rng.normal(size=n),
        }
    )
    _, _, spec = build_design("y ~ C(g) + x", train)

    new_data = pd.DataFrame({"g": ["z"], "x": [0.1]})
    with pytest.raises(ValueError):
        apply_design(spec, new_data)


def test_nan_not_dropped_silently() -> None:
    """A missing value must raise, not silently reduce the row count."""
    data = pd.DataFrame({"y": [0, 1, 0, 1], "x1": [1.0, 2.0, np.nan, 4.0]})
    with pytest.raises(ValueError):
        build_design("y ~ x1", data)


def test_one_sided_formula_returns_no_y() -> None:
    """A one-sided formula must build with `y is None` and the same X as the two-sided case."""
    rng = np.random.default_rng(6)
    n = 60
    data = pd.DataFrame(
        {
            "x1": rng.normal(size=n),
            "g": rng.choice(["a", "b", "c"], size=n),
        }
    )
    y_one, X_one, _ = build_design("~ x1 + C(g)", data)
    assert y_one is None

    data_two_sided = data.copy()
    data_two_sided["y"] = rng.binomial(1, 0.5, n)
    _, X_two, _ = build_design("y ~ x1 + C(g)", data_two_sided)

    assert list(X_one.columns) == list(X_two.columns)
    pd.testing.assert_frame_equal(X_one, X_two)


def test_variable_name_falls_back_to_str_without_root() -> None:
    """A formulaic variable without ``.root`` falls back to its own text."""

    class WithoutRoot(str):
        pass

    assert _variable_name(WithoutRoot("x1")) == "x1"

    class WithRoot(str):
        root = "df"

    assert _variable_name(WithRoot("df.x")) == "df"


def test_bool_response_same_as_int_response() -> None:
    """A boolean response column must produce the same design and fit as an int one."""
    rng = np.random.default_rng(0)
    n = 200
    x = rng.normal(size=n)
    y = rng.binomial(1, 1 / (1 + np.exp(-x)))
    df_int = pd.DataFrame({"y": y, "x": x})
    df_bool = pd.DataFrame({"y": y.astype(bool), "x": x})
    y_int, _, _ = build_design("y ~ x", df_int)
    y_bool, _, _ = build_design("y ~ x", df_bool)
    assert np.array_equal(y_bool.to_numpy(dtype=float), y_int.to_numpy(dtype=float))
    fit_int = fit_glm_formula("y ~ x", df_int, sm.families.Binomial())
    fit_bool = fit_glm_formula("y ~ x", df_bool, sm.families.Binomial())
    np.testing.assert_allclose(fit_bool.params.to_numpy(), fit_int.params.to_numpy(), rtol=1e-8)


def test_string_response_raises_value_error() -> None:
    """A string response column must raise ValueError instead of being encoded."""
    df = pd.DataFrame({"y": ["a", "b"] * 50, "x": np.arange(100, dtype=float)})
    with pytest.raises(ValueError):
        build_design("y ~ x", df)


def test_predict_na_action_nan() -> None:
    """``na_action="nan"`` predicts complete rows and returns NaN for missing predictors."""
    rng = np.random.default_rng(7)
    n = 40
    data = pd.DataFrame(
        {"y": rng.normal(size=n), "x1": rng.normal(size=n), "x2": rng.normal(size=n)}
    )
    fit = fit_glm_formula("y ~ x1 + x2", data, sm.families.Gaussian())

    new_data = data.copy()
    new_data.loc[3, "x1"] = np.nan
    new_data.loc[7, "x1"] = np.nan

    pred = fit.predict(new_data, na_action="nan")

    assert len(pred) == len(new_data)
    assert list(pred.index) == list(new_data.index)
    values = pred.to_numpy(dtype=float)
    assert list(np.nonzero(np.isnan(values))[0]) == [3, 7]

    reference = fit.predict(new_data.dropna())
    np.testing.assert_allclose(values[~np.isnan(values)], reference.to_numpy())


def test_predict_na_action_nan_duplicate_index() -> None:
    """``na_action="nan"`` keeps the length and duplicated index of ``new_data``."""
    rng = np.random.default_rng(11)
    n = 40
    data = pd.DataFrame(
        {"y": rng.normal(size=n), "x1": rng.normal(size=n), "x2": rng.normal(size=n)}
    )
    fit = fit_glm_formula("y ~ x1 + x2", data, sm.families.Gaussian())

    new_data = data.copy()
    new_data.loc[3, "x1"] = np.nan
    new_data.loc[7, "x1"] = np.nan
    new_data.index = pd.Index([i // 2 for i in range(n)])

    pred = fit.predict(new_data, na_action="nan")

    assert len(pred) == len(new_data)
    assert list(pred.index) == list(new_data.index)
    values = pred.to_numpy(dtype=float)
    assert list(np.nonzero(np.isnan(values))[0]) == [3, 7]

    reference = fit.predict(new_data.dropna())
    np.testing.assert_allclose(values[~np.isnan(values)], reference.to_numpy())


def test_predict_na_action_nan_categorical() -> None:
    """``na_action="nan"`` also works when the formula has a categorical term."""
    rng = np.random.default_rng(8)
    n = 40
    data = pd.DataFrame(
        {
            "y": rng.normal(size=n),
            "g": rng.choice(["a", "b", "c"], size=n),
            "x": rng.normal(size=n),
        }
    )
    fit = fit_glm_formula("y ~ C(g) + x", data, sm.families.Gaussian())

    new_data = data.copy()
    new_data.loc[3, "x"] = np.nan
    new_data.loc[7, "x"] = np.nan

    pred = fit.predict(new_data, na_action="nan")

    assert len(pred) == len(new_data)
    assert list(pred.index) == list(new_data.index)
    values = pred.to_numpy(dtype=float)
    assert list(np.nonzero(np.isnan(values))[0]) == [3, 7]

    reference = fit.predict(new_data.dropna())
    np.testing.assert_allclose(values[~np.isnan(values)], reference.to_numpy())


def test_predict_default_raises_on_nan() -> None:
    """The default ``na_action`` must keep raising on missing predictors."""
    rng = np.random.default_rng(9)
    n = 40
    data = pd.DataFrame(
        {"y": rng.normal(size=n), "x1": rng.normal(size=n), "x2": rng.normal(size=n)}
    )
    fit = fit_glm_formula("y ~ x1 + x2", data, sm.families.Gaussian())

    new_data = data.copy()
    new_data.loc[3, "x1"] = np.nan

    with pytest.raises(ValueError):
        fit.predict(new_data)


def test_predict_invalid_na_action() -> None:
    """An unknown ``na_action`` must raise ValueError."""
    rng = np.random.default_rng(10)
    n = 20
    data = pd.DataFrame(
        {"y": rng.normal(size=n), "x1": rng.normal(size=n), "x2": rng.normal(size=n)}
    )
    fit = fit_glm_formula("y ~ x1 + x2", data, sm.families.Gaussian())

    with pytest.raises(ValueError):
        fit.predict(data, na_action="foo")


def test_get_formula_variables_includes_column_named_C() -> None:
    """A column named C used as a plain term or as the response is reported."""
    assert get_formula_variables("L2 ~ C + t0") == ["C", "L2", "t0"]
    assert get_formula_variables("C ~ A + L") == ["A", "C", "L"]
    assert get_formula_variables("y ~ C(g) + x") == ["g", "x", "y"]
