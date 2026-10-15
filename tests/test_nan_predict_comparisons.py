"""The comparison step tolerates rows with missing predictors in the censor model.

When an id has no ``t0 == 0`` row, the package-built ``lag1_*`` columns are NaN
in that id's first remaining row. The censor model below uses ``lag1_A``, so the
comparison step must predict on such rows without raising.
"""
import numpy as np
import pandas as pd

from pygformula import ParametricGformula
from pygformula.interventions import static


def test_censor_model_with_ids_not_starting_at_time_zero() -> None:
    """Run the full pipeline when some ids have no ``t0 == 0`` row."""
    rng = np.random.default_rng(0)
    n_ids = 500
    n_times = 5
    n_rows = n_ids * n_times

    data = pd.DataFrame(
        {
            "id": np.repeat(np.arange(n_ids), n_times),
            "t0": np.tile(np.arange(n_times), n_ids),
            "L": rng.normal(size=n_rows),
            "A": rng.binomial(1, 0.4, n_rows),
            "Y": rng.binomial(1, 0.05, n_rows),
            "C": rng.binomial(1, 0.1, n_rows),
        }
    )

    # Drop the t0 == 0 row for about 10% of the ids, so their lag1_* columns
    # become NaN in their first remaining row.
    drop_ids = rng.choice(n_ids, size=int(0.1 * n_ids), replace=False)
    drop_mask = data["id"].isin(drop_ids) & (data["t0"] == 0)
    data = data.loc[~drop_mask].reset_index(drop=True)

    time_points = n_times
    g = ParametricGformula(
        obs_data=data,
        id="id",
        time_name="t0",
        outcome_name="Y",
        ymodel="Y ~ A + L",
        time_points=time_points,
        int_descript=["Never treat"],
        Intervention1_A=[static, np.zeros(time_points)],
        covnames=["L", "A"],
        covtypes=["normal", "binary"],
        covmodels=["L ~ lag1_L + A + t0", "A ~ lag1_L + L + t0"],
        outcome_type="survival",
        censor_name="C",
        censor_model="C ~ A + lag1_A",
        nsamples=0,
        seed=0,
    )
    g.fit()

    assert np.isfinite(np.asarray(g.g_results, dtype=float)).all()
