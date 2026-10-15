import re

import numpy as np
import pandas as pd
import pytest
import statsmodels.api as sm
import statsmodels.formula.api as smf
from scipy.interpolate import BSpline

from pygformula.utils.formula import apply_design, build_design, get_formula_variables


# --- Test Data Setup ---
@pytest.fixture(scope="module")
def spline_data() -> pd.DataFrame:
    """Generates training data for B-spline tests."""
    rng = np.random.default_rng(42)
    n = 300
    x = rng.uniform(0, 10, n)
    # y is a smooth function of x plus noise
    y = 5 + 0.2 * x - 0.001 * x**2 + rng.normal(0, 1, n)
    return pd.DataFrame({"x": x, "y": y})


# --- Basic spline behaviour ---


def test_bs_matches_patsy_training(spline_data: pd.DataFrame) -> None:
    formula = "y ~ bs(x, df=4)"
    y_pyg, X_pyg, _ = build_design(formula, spline_data)
    ours = sm.GLM(y_pyg, X_pyg, family=sm.families.Gaussian()).fit()
    ref = smf.glm(formula, spline_data, family=sm.families.Gaussian()).fit()
    np.testing.assert_allclose(
        ours.fittedvalues.to_numpy(), ref.fittedvalues.to_numpy(), rtol=0, atol=1e-8
    )


def test_bs_matches_scipy_basis(spline_data: pd.DataFrame) -> None:
    """Compare the formulaic B-spline columns with an independent scipy basis."""
    x = spline_data["x"].to_numpy()
    degree = 3
    # df=4, degree=3, include_intercept=False -> one interior knot (the median).
    interior = np.quantile(x, [0.5])
    knots = np.r_[[x.min()] * (degree + 1), interior, [x.max()] * (degree + 1)]
    n_basis = len(knots) - degree - 1  # 5 basis functions
    scipy_basis = BSpline(knots, np.eye(n_basis), degree)(x)

    _, X, _ = build_design("y ~ bs(x, df=4)", spline_data)
    # Convention (same as patsy): the spline basis drops its first column
    # (include_intercept=False) and the Intercept column takes its place.
    assert X.shape[1] == 5
    np.testing.assert_allclose(X.iloc[:, 0].to_numpy(), 1.0)
    np.testing.assert_allclose(X.iloc[:, 1:].to_numpy(), scipy_basis[:, 1:], rtol=0, atol=1e-10)


def test_bs_prediction_uses_training_knots(spline_data: pd.DataFrame) -> None:
    formula = "y ~ bs(x, df=4)"
    y, X_train, spec = build_design(formula, spline_data)
    new_df = pd.DataFrame({"x": [1.5, 3.2, 5.0, 7.7, 9.1]})  # inside training range

    X_new = apply_design(spec, new_df)
    ours = sm.GLM(y, X_train, family=sm.families.Gaussian()).fit()
    pred_ours = X_new.to_numpy() @ ours.params.to_numpy()
    ref = smf.glm(formula, spline_data, family=sm.families.Gaussian()).fit()
    np.testing.assert_allclose(pred_ours, ref.predict(new_df).to_numpy(), rtol=0, atol=1e-8)

    # One-row application must equal the same row of the full application
    for i in range(len(new_df)):
        row = apply_design(spec, new_df.iloc[[i]])
        np.testing.assert_allclose(row.to_numpy()[0], X_new.to_numpy()[i], atol=1e-12)


@pytest.mark.parametrize("value", [-5.0, 15.0])
def test_bs_out_of_range_behavior(spline_data: pd.DataFrame, value: float) -> None:
    """Values outside the training range raise a ValueError (never silent NaN).

    By default ``bs`` refuses to extrapolate and raises ("Some field values
    extend beyond upper and/or lower bounds"). ``apply_design`` wraps that error
    in a ValueError whose message contains the offending factor, ``bs(x, df=4)``.
    ``bs`` accepts an ``extrapolation`` argument, but the default is kept so
    that out-of-range values fail loudly.
    """
    _, _, spec = build_design("y ~ bs(x, df=4)", spline_data)
    new_df = pd.DataFrame({"x": [value]})
    with pytest.raises(ValueError, match=re.escape("bs(x, df=4)")):
        apply_design(spec, new_df)


def test_poly_and_mixed_terms(spline_data: pd.DataFrame) -> None:
    data = spline_data.copy()  # do not mutate the module-scoped fixture
    rng = np.random.default_rng(7)
    data["g"] = rng.choice(["a", "b"], len(data))

    y, X, spec = build_design("y ~ poly(x, 2) + C(g) + bs(x, df=3)", data)
    assert y is not None
    # Intercept + poly(x, 2) [2] + C(g) [1] + bs(x, df=3) [3]
    assert X.shape[1] == 7
    X_again = apply_design(spec, data)
    assert list(X_again.columns) == list(X.columns)
    np.testing.assert_allclose(X_again.to_numpy(), X.to_numpy(), rtol=0, atol=1e-12)


def test_spline_variable_names() -> None:
    formula = "y ~ bs(lag1_L2, df=4) + C(lag1_L) + t0"
    assert get_formula_variables(formula) == ["lag1_L", "lag1_L2", "t0", "y"]


# --- Extrapolation options ---


def test_bs_out_of_range_error_mentions_extrapolation(spline_data: pd.DataFrame) -> None:
    """The out-of-range error tells the user how to choose a behaviour."""
    _, _, spec = build_design("y ~ bs(x, df=4)", spline_data)
    new_df = pd.DataFrame({"x": [-5.0, 15.0]})
    # formulaic's message mentions "extrapolation" but not "clip", so this checks our own hint.
    with pytest.raises(ValueError, match=r"extrapolation='clip'"):
        apply_design(spec, new_df)


def test_bs_extrapolation_clip_keeps_boundary_values(spline_data: pd.DataFrame) -> None:
    """``extrapolation='clip'`` clamps out-of-range values to the training bounds."""
    _, _, spec = build_design("y ~ bs(x, df=4, extrapolation='clip')", spline_data)

    X_new = apply_design(spec, pd.DataFrame({"x": [-5.0, 15.0]}))
    X_min = apply_design(spec, pd.DataFrame({"x": [spline_data["x"].min()]}))
    X_max = apply_design(spec, pd.DataFrame({"x": [spline_data["x"].max()]}))

    np.testing.assert_allclose(
        X_new.iloc[0].to_numpy(), X_min.iloc[0].to_numpy(), rtol=0, atol=1e-12
    )
    np.testing.assert_allclose(
        X_new.iloc[1].to_numpy(), X_max.iloc[0].to_numpy(), rtol=0, atol=1e-12
    )


def test_bs_extrapolation_clip_inside_range_unchanged(spline_data: pd.DataFrame) -> None:
    """Inside the training range, 'clip' matches the default formula values.

    The column labels differ because formulaic embeds the full factor string
    (``extrapolation='clip'``) in each spline column name, but the columns line
    up one-to-one with the default design.
    """
    _, _, clip_spec = build_design("y ~ bs(x, df=4, extrapolation='clip')", spline_data)
    _, _, default_spec = build_design("y ~ bs(x, df=4)", spline_data)

    rng = np.random.default_rng(0)
    inside = pd.DataFrame({"x": rng.uniform(0, 10, 10)})

    X_clip = apply_design(clip_spec, inside)
    X_default = apply_design(default_spec, inside)

    assert len(X_clip.columns) == len(X_default.columns)
    np.testing.assert_allclose(X_clip.to_numpy(), X_default.to_numpy(), rtol=0, atol=1e-12)


def test_bs_extrapolation_na_still_rejected(spline_data: pd.DataFrame) -> None:
    """``extrapolation='na'`` still fails in apply_design because NaN is rejected."""
    _, _, spec = build_design("y ~ bs(x, df=4, extrapolation='na')", spline_data)
    with pytest.raises(ValueError):
        apply_design(spec, pd.DataFrame({"x": [15.0]}))
