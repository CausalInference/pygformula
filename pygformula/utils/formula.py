"""Formula layer built on top of `formulaic`.

This module replaces the implicit use of patsy (through the statsmodels
formula API) with explicit design-matrix construction using
`formulaic`. It provides three building blocks used by the rest of the
package:

- `get_formula_variables`: list the raw data column names a formula needs.
- `build_design`: build `(y, X, spec)` from a formula and a training DataFrame.
- `apply_design`: re-apply a stored `ModelSpec` to new data (for prediction).

formulaic API used (verified on formulaic 1.1.1 (Python 3.8) and 1.2.2):

- ``formulaic.Formula`` (constructor from an R-style string).
- ``Formula.required_variables`` (a ``set`` of ``Variable`` objects, each a
  ``str`` subclass; in formulaic 1.2.x it has a ``.root`` property, in 1.1.x
  it does not (handled by `_variable_name`)).
- ``formulaic.model_matrix(formula, data, na_action=...)`` returning either a
  ``formulaic.model_matrix.ModelMatrix`` (one-sided formula) or a
  ``formulaic.model_matrix.ModelMatrices`` structured object exposing
  ``.lhs`` and ``.rhs`` (two-sided formula).
- ``ModelMatrix.model_spec`` to retrieve the fitted ``formulaic.ModelSpec``.
- ``ModelSpec.get_model_matrix(new_data, na_action=...)`` to re-apply a
  stored spec to new data.
- ``formulaic.errors.DataMismatchWarning``, raised as a warning by
  `ModelSpec.get_model_matrix` when new data contains unseen categorical
  levels. This module turns it into a `ValueError` with `warnings.catch_warnings`.

Note: formulaic's ``Formula.required_variables`` does not report variables
used inside stateful transforms such as ``bs``, ``poly`` and ``center``.
`get_formula_variables` adds them by parsing each factor expression.
"""

import ast
import warnings
from typing import Any, Iterator, List, Optional, Set, Tuple

import numpy as np
import pandas as pd
import statsmodels.api as sm
from formulaic import Formula, model_matrix
from formulaic.errors import DataMismatchWarning
from formulaic.model_spec import ModelSpec


def _names_in_expr(expr: str) -> Set[str]:
    """Return the data variable names used in a Python factor expression.

    Args:
        expr: The Python expression of a formulaic factor, for example
            ``bs(x, df=4)``.

    Returns:
        The names used as data variables. Called functions (``bs``) and
        attribute bases (``np`` in ``np.log(x)``) are excluded.
    """
    tree = ast.parse(expr, mode="eval")
    excluded: Set[str] = set()
    for node in ast.walk(tree):
        if isinstance(node, ast.Call) and isinstance(node.func, ast.Name):
            excluded.add(node.func.id)
        elif isinstance(node, ast.Attribute):
            base = node.value
            while isinstance(base, ast.Attribute):
                base = base.value
            if isinstance(base, ast.Name):
                excluded.add(base.id)
    return {
        node.id for node in ast.walk(tree) if isinstance(node, ast.Name) and node.id not in excluded
    }


def _iter_terms(parsed: Formula) -> Iterator[Any]:
    """Yield every term of a parsed formula, left- and right-hand side.

    Args:
        parsed: A parsed formulaic ``Formula``.

    Returns:
        An iterator over the formulaic terms.
    """
    found = False
    for side in ("lhs", "rhs"):
        try:
            part = getattr(parsed, side)
        except AttributeError:
            continue
        if part is not None:
            found = True
            yield from part
    if not found:
        yield from parsed


def _variable_name(variable: object) -> str:
    """Return the root name of a formulaic variable, or its text.

    Args:
        variable: A formulaic ``Variable``. In formulaic 1.2.x it is a ``str``
            subclass exposing a ``.root`` property; in 1.1.x it is a plain
            ``str`` subclass without one.

    Returns:
        The ``.root`` attribute when present and non-empty, otherwise
        ``str(variable)``.
    """
    root = getattr(variable, "root", None)
    return root if root else str(variable)


def get_formula_variables(formula: str) -> List[str]:
    """Return the sorted list of raw data column names required by a formula.

    This resolves `C(x)`, `bs(x, df=4)`, `I(x**2)`, `x:y`, `log(x)`, etc. to
    their underlying column names using formulaic's own parsing (no regex).

    Args:
        formula: An R-style formula string, one-sided (`~ x1 + x2`) or
            two-sided (`y ~ x1 + x2`).

    Returns:
        The sorted list of unique raw column names required by the formula
        (LHS and RHS, when present).

    Raises:
        ValueError: If the formula cannot be parsed by formulaic.
    """
    try:
        parsed = Formula(formula)
    except Exception as exc:  # re-raised as a clear ValueError
        raise ValueError(f"Could not parse formula {formula!r}: {exc}") from exc

    names = {_variable_name(variable) for variable in parsed.required_variables}
    # formulaic's required_variables omits variables used inside stateful
    # transforms (bs, poly, center, scale, ...). Recover them from the factor
    # expressions so that missing-value checks cover every needed column.
    for term in _iter_terms(parsed):
        for factor in term.factors:
            if factor.eval_method.value == "python":
                names |= _names_in_expr(factor.expr)
            elif factor.eval_method.value == "lookup" and factor.expr.isidentifier():
                # A plain column that formulaic omits from required_variables,
                # for example a column named C (same name as the C() transform).
                names.add(factor.expr)
    return sorted(names)


def drop_rows_with_missing(formula: str, data: pd.DataFrame, label: str) -> pd.DataFrame:
    """Drop rows with missing values in the formula's variables, with a warning.

    `smf.glm` used to drop rows whose predictors contained NaN silently. The
    formulaic design layer is strict and raises instead, so the fit helpers
    call this before fitting to reproduce the old behaviour with an explicit
    warning. `build_design` and `apply_design` remain strict.

    Args:
        formula: An R-style formula string whose variables are checked.
        data: The DataFrame to filter. It is never modified.
        label: A short human-readable name of the model being fit, used in the
            warning message.

    Returns:
        ``data`` unchanged (the same object) when the formula has no variables
        present in ``data`` or none of them contain missing values; otherwise a
        new DataFrame with the rows containing missing values removed.

    """
    cols = [c for c in get_formula_variables(formula) if c in data.columns]
    if not cols:
        return data
    mask = data[cols].isna().any(axis=1)
    n = int(mask.sum())
    if n == 0:
        return data
    counts = {c: int(data[c].isna().sum()) for c in cols if data[c].isna().any()}
    warnings.warn(
        f"{label}: {n} of {len(data)} rows have missing values {counts} and were "
        "dropped before fitting (earlier versions dropped them silently).",
        UserWarning,
        stacklevel=2,
    )
    return data.loc[~mask]


def build_design(
    formula: str, data: pd.DataFrame
) -> Tuple[Optional[pd.Series], pd.DataFrame, ModelSpec]:
    """Build the design matrices for a formula on training data.

    Args:
        formula: An R-style formula string, one-sided (`~ x1 + x2`, no `y`)
            or two-sided (`y ~ x1 + x2`).
        data: The DataFrame to materialize the formula against. The caller is
            responsible for dropping rows with missing values beforehand;
            this function never silently drops rows.

    Returns:
        A tuple `(y, X, spec)` where `y` is the LHS as a `pandas.Series` (or
        `None` if the formula is one-sided), `X` is the RHS as a DataFrame
        with named columns (including `Intercept`), and `spec` is the
        formulaic `ModelSpec` for the RHS, to be reused with `apply_design`.

    Raises:
        ValueError: If the formula cannot be parsed, if any row used to
            build the design has a missing value, or if the response is not
            numeric or boolean (the left-hand side produces more than one
            column).
    """
    try:
        parsed = Formula(formula)
    except Exception as exc:  # re-raised as a clear ValueError
        raise ValueError(f"Could not parse formula {formula!r}: {exc}") from exc

    try:
        result = model_matrix(parsed, data, na_action="raise")
    except Exception as exc:  # re-raised as a clear ValueError
        raise ValueError(
            f"Could not build the design matrix for formula {formula!r}: {exc}"
        ) from exc

    if hasattr(result, "lhs"):
        y_matrix = result.lhs
        if y_matrix is not None and y_matrix.shape[1] != 1:
            raise ValueError(f"The response of formula {formula!r} must be numeric or boolean; "
                             f"it produced {y_matrix.shape[1]} columns.")
        y = None if y_matrix is None else y_matrix.iloc[:, 0]
        X = result.rhs
        spec = X.model_spec
    else:
        y = None
        X = result
        spec = X.model_spec

    return y, X, spec


def apply_design(spec: ModelSpec, new_data: pd.DataFrame, na_action: str = "raise") -> pd.DataFrame:
    """Build the RHS design matrix for new data using a stored `ModelSpec`.

    The stored spec (levels, knots, etc.) is reused so that the returned
    matrix has the same columns as the one `build_design` produced at fit
    time. The row count and index of `new_data` are preserved.

    Args:
        spec: A `formulaic.ModelSpec` previously returned by `build_design`.
        new_data: The new DataFrame to materialize the RHS formula against.
        na_action: How to treat rows whose required columns contain missing
            values. ``"raise"`` (default) raises `ValueError`. ``"nan"`` keeps
            those rows but leaves their design-matrix cells as NaN.

    Returns:
        A DataFrame with the same columns as at fit time, indexed like
        `new_data`.

    Raises:
        ValueError: If `na_action` is not ``"raise"`` or ``"nan"``, if
            `new_data` has missing values in the required columns (with
            ``na_action="raise"``), or if it contains categorical levels that
            were not seen when `spec` was built.
    """
    if na_action == "raise":
        formulaic_action = "raise"
    elif na_action == "nan":
        formulaic_action = "ignore"
    else:
        raise ValueError(f"Unknown na_action {na_action!r}; expected 'raise' or 'nan'.")

    try:
        with warnings.catch_warnings():
            warnings.simplefilter("error", DataMismatchWarning)
            X = spec.get_model_matrix(new_data, na_action=formulaic_action)
    except DataMismatchWarning as exc:
        raise ValueError(f"New data contains categorical levels unseen at fit time: {exc}") from exc
    except Exception as exc:  # re-raised as a clear ValueError
        message = f"Could not apply the stored design to new data: {exc}"
        if "bs(" in str(
            spec.formula
        ) and "Some field values extend beyond upper and/or lower bounds" in str(exc):
            message += (
                " A spline term received values outside the range seen when "
                "fitting. Values are not clipped silently. To choose a "
                "behaviour, write the term with an explicit extrapolation "
                "option, for example bs(x, df=4, extrapolation='clip') to keep "
                "the boundary value."
            )
        raise ValueError(message) from exc

    if len(X) != len(new_data):
        raise ValueError(
            "Applying the stored design dropped rows: expected "
            f"{len(new_data)} rows, got {len(X)}. "
            "This usually means new_data contains missing values."
        )

    return X


class GLMResult:
    """Result of a GLM fitted on a formulaic design matrix.

    Keeps the formulaic `ModelSpec` so that predictions on new data reuse the
    knots and categorical levels learned at fit time. Any attribute not
    defined here is delegated to the underlying statsmodels result.

    Attributes:
        _result: The underlying statsmodels GLM results object.
        _spec: The formulaic `ModelSpec` used to build the training design.
        _formula: The original formula string.
    """

    def __init__(self, result: Any, spec: ModelSpec, formula: str) -> None:
        """Initialize the result object.

        Args:
            result: The underlying statsmodels GLM results object.
            spec: The formulaic `ModelSpec` returned by `build_design`.
            formula: The original formula string.
        """
        self._result = result
        self._spec = spec
        self._formula = formula

    @property
    def params(self) -> pd.Series:
        """The fitted parameters, indexed by design-matrix column name."""
        return self._result.params

    @property
    def bse(self) -> pd.Series:
        """The standard errors of the fitted parameters."""
        return self._result.bse

    def cov_params(self) -> pd.DataFrame:
        """Return the variance-covariance matrix of the parameters.

        Returns:
            The variance-covariance matrix of the fitted parameters.
        """
        return self._result.cov_params()

    def summary(self) -> Any:
        """Return the statsmodels summary of the fit.

        Returns:
            The statsmodels `Summary` object (use `str()` to print it).
        """
        return self._result.summary()

    def predict(
        self,
        new_data: Optional[pd.DataFrame] = None,
        na_action: str = "raise",
    ) -> pd.Series:
        """Predict on new data, or return the fitted values.

        Args:
            new_data: The DataFrame to predict on. If `None`, the fitted
                values on the training data are returned.
            na_action: How to treat rows whose predictors contain missing
                values. ``"raise"`` (default) raises `ValueError`. ``"nan"``
                returns NaN for those rows and predicts the remaining rows
                normally, keeping the length and index of `new_data`.

        Returns:
            A Series named `"prediction"` and indexed like `new_data`, or a
            Series named `"fitted"` when `new_data` is `None`.

        Raises:
            ValueError: If `na_action` is not ``"raise"`` or ``"nan"``, or (with
                ``na_action="raise"``) if `new_data` has missing values, unseen
                categorical levels, or values outside the range of a spline term.
        """
        if new_data is None:
            fitted = self._result.fittedvalues
            if isinstance(fitted, pd.Series):
                return fitted.rename("fitted")
            return pd.Series(fitted, name="fitted")

        if na_action not in ("raise", "nan"):
            raise ValueError(f"Unknown na_action {na_action!r}; expected 'raise' or 'nan'.")

        work = new_data.reset_index(drop=True)
        X_new = apply_design(self._spec, work, na_action=na_action)
        values = np.asarray(self._result.predict(X_new.to_numpy()))
        return pd.Series(values, index=new_data.index, name="prediction")

    def __getattr__(self, name: str) -> Any:
        """Delegate unknown attributes to the underlying statsmodels result.

        Args:
            name: The attribute name.

        Returns:
            The attribute of the underlying statsmodels result.

        Raises:
            AttributeError: If the name is private or not found.
        """
        # Private names are never delegated: this also avoids infinite
        # recursion when `_result` is not set yet (copy, pickle).
        if name.startswith("_"):
            raise AttributeError(name)
        return getattr(self._result, name)


def fit_glm_formula(formula: str, data: pd.DataFrame, family: Any) -> GLMResult:
    """Fit a GLM from an R-style formula using formulaic for the design.

    Replaces `smf.glm(formula, data, family).fit()` without using patsy.
    Unused levels of categorical predictors are removed before the design is
    built, so predicting a row with such a level raises ValueError.

    Args:
        formula: A two-sided R-style formula string.
        data: The DataFrame to materialize the formula against.
        family: A statsmodels GLM family instance, for example
            `sm.families.Binomial()`.

    Returns:
        A `GLMResult` holding the fitted model and the design `ModelSpec`.

    Raises:
        ValueError: If the formula is one-sided or cannot be built, or if the
            fitted parameters or the log-likelihood are not finite.
    """
    # Work on a copy of the data so the caller's frame is never modified.
    # Categorical predictors keep their dtype but drop categories with no row,
    # so an unused level does not produce an all-zero dummy column and break
    # the rank check.
    variables = get_formula_variables(formula)
    data_copy = data.copy()
    for col in variables:
        if col in data_copy and isinstance(data_copy[col].dtype, pd.CategoricalDtype):
            data_copy[col] = data_copy[col].cat.remove_unused_categories()

    y, X, spec = build_design(formula, data_copy)
    if y is None:
        raise ValueError(f"Formula {formula!r} has no response variable.")
    result = sm.GLM(y, X, family=family).fit()
    # statsmodels may report convergence with NaN estimates: check explicitly.
    if not np.all(np.isfinite(np.asarray(result.params))) or not np.isfinite(result.llf):
        raise ValueError(f"GLM fit for formula {formula!r} produced non-finite estimates.")
    return GLMResult(result, spec, formula)
