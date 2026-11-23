"""Multinomial (categorical response) regression wrapper.

This module fits a multinomial logistic regression for a categorical
covariate. It reuses the formulaic design layer (:mod:`pygformula.utils.formula`)
so that splines, ``C()`` terms and other formula features work, and it wraps the
statsmodels ``MNLogit`` estimator with a checked fallback chain, because the
default Newton optimiser can diverge silently (reporting ``converged = True``
while every parameter is NaN).

The two failure modes this module guards against are:

- A fit whose parameters or log-likelihood are not finite, even though
  statsmodels reports convergence. Each attempt is accepted only when the
  parameters and the log-likelihood are finite (and, except for the regularized
  last resort, when ``converged`` is also true).
- Category labels built from a different set of rows than the codes. The
  response categories are always computed from the rows actually passed in
  (``sorted(pd.unique(...))``), never from a ``pd.Categorical`` dtype, which may
  keep categories that have no row in the fit.
"""

import warnings
from typing import Any, Dict, List, Optional, Tuple

import numpy as np
import pandas as pd
import statsmodels.api as sm
from numpy.linalg import LinAlgError

from pygformula.utils.formula import apply_design, build_design, get_formula_variables

# L1 penalty used by the last-resort regularized fallback. Chosen on a
# quasi-separated synthetic case (an empty (lag, outcome) cell,
# near-deterministic transitions and an unscaled predictor in [18, 89]; see
# test_multinomial_quasi_separation_contract). It is small enough to barely
# bias the estimates while still regularizing away the divergence.
_REGULARIZED_ALPHA = 1e-4


def _stable_softmax(logits: np.ndarray) -> np.ndarray:
    """Apply a numerically stable softmax to each row of ``logits``.

    The row maximum is subtracted before exponentiating, so large linear
    predictors (for example 1000) do not overflow to infinity.

    Args:
        logits: A 1-D array, or a 2-D array with one row per observation.

    Returns:
        An array of the same shape whose rows are finite and sum to 1.
    """
    arr = np.asarray(logits, dtype=float)
    single = arr.ndim == 1
    if single:
        arr = arr.reshape(1, -1)
    shifted = arr - np.max(arr, axis=1, keepdims=True)
    exp = np.exp(shifted)
    probs = exp / np.sum(exp, axis=1, keepdims=True)
    return probs[0] if single else probs


class MultinomialResult:
    """Result of a multinomial logistic regression on a formulaic design matrix.

    Keeps the formulaic ``ModelSpec`` so predictions on new data reuse the knots
    and categorical levels learned at fit time, and keeps the observed response
    categories so predicted columns are labelled correctly even when the labels
    are non-contiguous or a category is absent from some rows.

    ``params`` has shape ``(n_predictors, K - 1)``: the first category in
    ``categories`` is the reference level, whose coefficients are fixed at zero
    and which therefore has no column; column ``j`` belongs to
    ``categories[j + 1]``.

    Attributes:
        categories: The sorted observed categories of the response. The first
            entry is the reference level.
        spec: The formulaic ``ModelSpec`` of the RHS design.
        fit_method: The name of the method that produced the accepted fit
            (``"newton"``, ``"bfgs"``, ``"lbfgs"`` or ``"l1"``).
        attempts: A list of dicts, one per fitting attempt, each with the keys
            ``method``, ``converged``, ``finite`` and ``message``.
    """

    def __init__(
        self,
        result: Any,
        categories: List[Any],
        spec: Any,
        fit_method: str,
        attempts: List[Dict[str, Any]],
        formula: str,
        name: str,
    ) -> None:
        """Initialize the result object.

        Args:
            result: The underlying statsmodels multinomial results object.
            categories: The sorted observed response categories.
            spec: The formulaic ``ModelSpec`` returned by ``build_design``.
            fit_method: The name of the accepted fitting method.
            attempts: The list of per-attempt diagnostic dicts.
            formula: The original formula string.
            name: The covariate (response) name used in messages.
        """
        self._result = result
        self.categories = list(categories)
        self.spec = spec
        self.fit_method = fit_method
        self.attempts = attempts
        self._formula = formula
        self._name = name

    @property
    def params(self) -> pd.DataFrame:
        """The fitted parameters as a ``(n_predictors, K - 1)`` DataFrame.

        The index is ``spec.column_names`` (the design columns) and the columns
        are ``categories[1:]`` (the non-reference categories). Column ``j``
        belongs to ``categories[j + 1]``; the reference category
        (``categories[0]``) has no column.
        """
        return pd.DataFrame(
            np.asarray(self._result.params, dtype=float),
            index=list(self.spec.column_names),
            columns=list(self.categories[1:]),
        )

    @property
    def bse(self) -> pd.DataFrame:
        """The standard errors of the fitted parameters, as a labelled DataFrame.

        The index and columns match :meth:`params`. When statsmodels cannot
        provide a covariance matrix (the regularized ``l1`` fallback stores
        none, and a Hessian that cannot be inverted gives none either), a
        DataFrame of NaN with the same labels is returned instead.
        """
        try:
            values = np.asarray(self._result.bse, dtype=float)
        except ValueError:
            values = np.full(np.asarray(self._result.params, dtype=float).shape, np.nan)
        return pd.DataFrame(
            values,
            index=list(self.spec.column_names),
            columns=list(self.categories[1:]),
        )

    def cov_params(self) -> Any:
        """Return the variance-covariance matrix of the parameters.

        Returns:
            The variance-covariance matrix of the fitted parameters. When
            statsmodels cannot provide a covariance matrix (the ``l1`` fallback
            stores none, and a Hessian that cannot be inverted gives none
            either), a square array of NaN is returned.
        """
        try:
            return self._result.cov_params()
        except ValueError:
            size = int(np.asarray(self._result.params).size)
            return np.full((size, size), np.nan)

    def summary(self) -> Any:
        """Return a summary of the fit.

        When statsmodels cannot provide a covariance matrix (the regularized
        ``l1`` fallback stores none, and a Hessian that cannot be inverted gives
        none either), the statsmodels summary would raise, so a plain-text
        summary is returned instead. Otherwise the statsmodels ``Summary``
        object is returned unchanged.

        Returns:
            The statsmodels ``Summary`` object, or a ``str`` when the covariance
            matrix is not available.
        """
        try:
            self._result.cov_params()
        except ValueError:
            return self._no_covariance_summary()
        return self._result.summary()

    def _no_covariance_summary(self) -> str:
        """Build a plain-text summary for a fit without a covariance matrix.

        Returns:
            A multi-line string with the covariate name, the fitting method,
            the categories, a note that standard errors are unavailable, and
            the parameter estimates as a DataFrame.
        """
        params = pd.DataFrame(
            self.params,
            index=list(self.spec.column_names),
            columns=list(self.categories[1:]),
        )
        lines = [
            f"Multinomial fit for covariate {self._name!r} (method {self.fit_method!r})",
            f"Categories (reference first): {self.categories}",
            (
                "Standard errors are not available because the covariance matrix "
                "could not be computed."
            ),
            "",
            params.to_string(),
        ]
        return "\n".join(lines)

    def predict(self, new_data: pd.DataFrame) -> pd.DataFrame:
        """Predict class probabilities for new rows.

        The stored ``ModelSpec`` is re-applied with ``apply_design``, the linear
        predictors for the ``K - 1`` non-reference classes are computed, a zero
        column is inserted for the reference class, and a numerically stable
        softmax turns the linear predictors into probabilities.

        Args:
            new_data: The DataFrame to predict on.

        Returns:
            A DataFrame of shape ``(n, K)`` whose columns are exactly
            ``categories`` and whose index is the index of ``new_data``. Each
            row sums to 1.

        Raises:
            ValueError: If ``new_data`` has missing values or unseen categorical
                levels (raised by ``apply_design``), or if the computed
                probabilities are not finite or do not sum to 1.
        """
        X_new = apply_design(self.spec, new_data)
        params = np.asarray(self._result.params, dtype=float)  # (p, K - 1)
        linear = np.asarray(X_new, dtype=float) @ params  # (n, K - 1)
        # Insert a zero column for the reference category, then softmax.
        logits = np.column_stack([np.zeros(len(X_new)), linear])  # (n, K)
        probs = _stable_softmax(logits)
        if not np.all(np.isfinite(probs)):
            raise ValueError(
                "Multinomial prediction produced non-finite probabilities for "
                f"formula {self._formula!r}."
            )
        if not np.allclose(np.sum(probs, axis=1), 1.0, atol=1e-8):
            raise ValueError(
                "Multinomial prediction produced rows that do not sum to 1 "
                f"for formula {self._formula!r}."
            )
        return pd.DataFrame(probs, index=new_data.index, columns=self.categories)

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


def _attempt_fit(
    model: Any,
    method: str,
    kwargs: Dict[str, Any],
    require_converged: bool,
    regularized: bool = False,
) -> Tuple[Optional[Any], Dict[str, Any]]:
    """Run one fitting attempt and record its diagnostics.

    Args:
        model: The ``sm.MNLogit`` model instance.
        method: The name recorded in the attempt dict.
        kwargs: Keyword arguments for ``model.fit`` (or ``model.fit_regularized``).
        require_converged: When True the attempt is accepted only if
            ``converged`` is also True; when False finite parameters and
            log-likelihood are enough.
        regularized: When True, call ``model.fit_regularized`` instead of
            ``model.fit``.

    Returns:
        A tuple ``(result, attempt)`` where ``result`` is the fitted result if
        the attempt is accepted and ``None`` otherwise, and ``attempt`` is a
        dict with keys ``method``, ``converged``, ``finite`` and ``message``.
    """
    attempt: Dict[str, Any] = {
        "method": method,
        "converged": False,
        "finite": False,
        "message": "",
    }
    try:
        with warnings.catch_warnings(record=True) as caught:
            warnings.simplefilter("always")
            if regularized:
                result = model.fit_regularized(**kwargs)
            else:
                result = model.fit(**kwargs)
        params = np.asarray(result.params, dtype=float)
        llf = float(result.llf)
        converged = bool(result.mle_retvals.get("converged", False))
        finite = bool(np.all(np.isfinite(params)) and np.isfinite(llf))
        attempt["converged"] = converged
        attempt["finite"] = finite
        if caught:
            attempt["message"] = "; ".join(str(w.message) for w in caught)
        if finite and (not require_converged or converged):
            return result, attempt
        return None, attempt
    except (LinAlgError, ValueError, FloatingPointError) as exc:
        attempt["message"] = str(exc)
        return None, attempt


def fit_multinomial_formula(
    formula: str, data: pd.DataFrame, name: Optional[str] = None
) -> MultinomialResult:
    """Fit a multinomial logistic regression from an R-style formula.

    The response (the left-hand side of the formula) must be a plain column
    name. The design is built with the formulaic layer from the right-hand side
    only, so splines and ``C()`` terms are supported. Categorical predictor
    columns have their unused categories removed before building the design, so
    an unused level does not create an empty dummy column and break the rank
    check.

    The fit tries Newton, BFGS and L-BFGS in that order, accepting the first
    attempt whose parameters and log-likelihood are finite (and whose
    ``converged`` flag is True), then falls back to an L1-regularized fit.
    Every attempt is wrapped in ``warnings.catch_warnings`` and a try/except for
    ``LinAlgError``, ``ValueError`` and ``FloatingPointError``.

    Args:
        formula: A two-sided R-style formula string such as
            ``L ~ C(lag1_L) + t0``.
        data: The DataFrame to fit on. The caller is responsible for dropping
            rows with missing values beforehand.
        name: The covariate name used in error and warning messages. Defaults to
            the response column name.

    Returns:
        A `MultinomialResult` holding the fit, the observed categories and the
        design spec.

    Raises:
        ValueError: If the left-hand side is not a plain column name, if the
            response has fewer than two observed categories, if the design
            matrix is not full column rank, or if every fitting attempt is
            rejected.
    """
    if "~" not in formula:
        raise ValueError(f"Formula {formula!r} has no '~' separating the response.")
    lhs, rhs = formula.split("~", 1)
    response = lhs.strip()
    if not response or not response.isidentifier():
        raise ValueError(
            f"Expected a plain column name on the left-hand side of {formula!r}, got {response!r}."
        )
    if name is None:
        name = response

    # Work on a copy of the columns the formula needs. Categorical predictors
    # keep their dtype but drop categories with no row, so an unused level does
    # not produce an all-zero dummy column.
    variables = get_formula_variables(formula)
    data_copy = data[variables].copy()
    for col in variables:
        if col == response:
            continue
        if isinstance(data_copy[col].dtype, pd.CategoricalDtype):
            data_copy[col] = data_copy[col].cat.remove_unused_categories()

    # Categories come from the rows actually passed in, never from the dtype.
    categories = sorted(pd.unique(data[response].dropna()))
    if len(categories) < 2:
        raise ValueError(
            f"Multinomial fit for covariate {name!r} requires at least two "
            f"categories, found {len(categories)}."
        )

    codes = pd.Index(categories).get_indexer(data[response])
    if (codes == -1).any():
        raise ValueError(
            f"Multinomial fit for covariate {name!r} encountered response values "
            f"outside the observed categories."
        )
    if not np.array_equal(np.sort(np.unique(codes)), np.arange(len(categories))):
        raise ValueError(
            f"Multinomial fit for covariate {name!r} produced a non-contiguous "
            f"set of response codes."
        )

    # One-sided RHS formula; build_design adds the Intercept and stores the spec.
    _, X, spec = build_design("~ " + rhs, data_copy)
    # Columns are scaled to unit norm for the rank check only, so that predictors
    # on very different scales (for example year and year ** 2) are not flagged
    # as collinear. The unscaled X is still the one passed to MNLogit.
    X_values = np.asarray(X, dtype=float)
    norms = np.linalg.norm(X_values, axis=0)
    rank = np.linalg.matrix_rank(X_values / np.where(norms > 0, norms, 1.0))
    if rank < X.shape[1]:
        raise ValueError(
            f"Design matrix for formula {formula!r} is not full column rank "
            f"(rank {rank} < {X.shape[1]} columns)."
        )

    model = sm.MNLogit(codes, X)

    attempts: List[Dict[str, Any]] = []
    result: Optional[Any] = None
    fit_method: Optional[str] = None

    fit_specs = [
        ("newton", {"method": "newton", "maxiter": 100, "disp": 0}, True),
        ("bfgs", {"method": "bfgs", "maxiter": 1000, "disp": 0}, True),
        ("lbfgs", {"method": "lbfgs", "maxiter": 5000, "disp": 0}, True),
    ]
    for method, kwargs, require_converged in fit_specs:
        res, attempt = _attempt_fit(model, method, kwargs, require_converged)
        attempts.append(attempt)
        if res is not None:
            result = res
            fit_method = method
            break

    if result is None:
        # Last resort: L1-regularized fit; finite params and llf are enough.
        res, attempt = _attempt_fit(
            model,
            "l1",
            {"method": "l1", "alpha": _REGULARIZED_ALPHA, "disp": 0},
            require_converged=False,
            regularized=True,
        )
        attempts.append(attempt)
        if res is not None:
            result = res
            fit_method = "l1"

    if result is None:
        details = "; ".join(
            f"{a['method']}: converged={a['converged']}, finite={a['finite']}, "
            f"message={a['message']!r}"
            for a in attempts
        )
        raise ValueError(
            f"Multinomial fit for covariate {name!r} failed with formula "
            f"{formula!r}. Attempts: {details}. Try a simpler formula or "
            f"merging rare categories."
        )

    try:
        bse_finite = bool(np.all(np.isfinite(np.asarray(result.bse, dtype=float))))
    except ValueError:
        # No covariance matrix: l1 results store none, and a Hessian that cannot
        # be inverted (quasi-separation) gives none for the other methods.
        bse_finite = False
    if not bse_finite:
        warnings.warn(
            f"Standard errors for the multinomial fit of covariate {name!r} are "
            f"not available or not all finite.",
            UserWarning,
            stacklevel=2,
        )

    if fit_method != "newton":
        warnings.warn(
            f"Multinomial fit for covariate {name!r} used method {fit_method!r} instead of Newton.",
            UserWarning,
            stacklevel=2,
        )

    return MultinomialResult(result, categories, spec, fit_method, attempts, formula, name)
