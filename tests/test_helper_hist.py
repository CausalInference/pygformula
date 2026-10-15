"""get_cov_hist_info extracts spline, C() and history terms correctly."""

from pygformula.utils.helper import get_cov_hist_info


def test_cov_hist_info_with_spline_terms() -> None:
    """History terms are extracted for bs(), C() and lag_cumavg formulas.

    The spline basis ``bs(...)`` wraps the lagged/cumavg terms, and ``C()``
    wraps the lagged binary covariate. ``get_cov_hist_info`` must still report
    the wrapped terms as history terms of the corresponding covariate.
    """
    cov_hist = get_cov_hist_info(
        covnames=["L2", "A"],
        covtypes=["normal", "binary"],
        covmodels=[
            "L2 ~ bs(lag1_L2, df=4) + lag_cumavg1_L2 + t0",
            "A ~ C(lag1_A) + L2",
        ],
        ymodel="Y ~ bs(cumavg_L2, df=3) + A",
    )

    assert cov_hist["L2"]["lagged"] == [["lag1_L2"], [1]]
    assert cov_hist["L2"]["cumavg"] == ["cumavg_L2"]
    assert cov_hist["L2"]["lagavg"] == [["lag_cumavg1_L2"], [1]]

    assert cov_hist["A"]["lagged"] == [["lag1_A"], [1]]
