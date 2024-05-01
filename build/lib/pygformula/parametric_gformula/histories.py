import numpy as np
import pandas as pd


def update_precoded_history(pool, covnames, cov_hist, covtypes, time_name, id, below_zero_indicator, baselags,
                   ts_visit_names=None):
    """
    This internal function is used to add new columns to the original pool for the three precoded historical terms (the
    lagged term, cumavg term, and lagavg term) in the model statement.

    Parameters
    ----------
    pool : DataFrame
        A DataFrame that contains the observed or simulated data up to the maximum time step of the data table.
        The historical terms at all time steps in the data table are to be updated.

    covnames : List
        A list of strings specifying the names of the time-varying covariates.

    cov_hist : Dict
        A dictionary whose keys are covariate names and values are sub-dictionaries with historical information for
        covariates. Each sub-dictionaty contains keys 'lagged', 'cumavg' and 'lagavg', the corresponding value for the
        key 'lagged' is a two-element list where the first element is a list with all lagged terms, the second element
        is a list with the corresponding lagged numbers. Same for the key 'lagavg'. The corresponding value for the key
        'cumavg' is a list with all cumavg terms.

    covtypes : List
        A list of strings specifying the type of each time-varying covariate included in covnames. The list must be
        the same length as covnames and in the same order. The supported types: “binary”, “normal”, “categorical”,
        “bounded normal”, “zero-inflated normal”, “truncated normal”, “absorbing”, “categorical time”, "square time"
        and "custom".

    time_name : Str
        A string specifying the name of the time variable in obs_data.

    id : Str
        A string specifying the name of the id variable in obs_data.

    below_zero_indicator : Bool
        A boolean variable indicating if the obs_data contains pre-baseline times.

    baselags : Bool
        A boolean value specifying the convention used for lagi and lag_cumavgi terms in the model statements when
        pre-baseline times are not included in obs_data and when the current time index, t, is such that t < i. If this
        argument is set to False, the value of all lagi and lag_cumavgi terms in this context are set to 0 (for
        non-categorical covariates) or the reference level (for categorical covariates). If this argument is set to
        True, the value of lagi and lag_cumavgi terms are set to their values at time 0.

    ts_visit_names : List
        A list of strings, each of which indicates the number of consecutive missed visits for one covariate before an
        individual is censored.

    Returns
    -------
    None : The original input pool has been updated and nothing is returned.

    """

    if ts_visit_names:
        covnames = covnames + ts_visit_names

    for k, cov in enumerate(covnames):
        if ts_visit_names is not None:
            cov_type = covtypes[k] if cov not in ts_visit_names else None
        else:
            cov_type = covtypes[k]

        lagged_covs = cov_hist[cov]['lagged'][0]
        lagged_nums = cov_hist[cov]['lagged'][1]
        if len(lagged_covs) > 0:  # create lag variable
            for i, lagged_cov in enumerate(lagged_covs):
                if cov_type == 'categorical':
                    if below_zero_indicator:
                        pool[lagged_cov] = np.array(pool.groupby([id])[cov].shift(lagged_nums[i]))
                    else:
                        fill_values = pool.groupby([id])[cov].transform('first') if baselags else \
                            pd.Categorical(pool[cov]).categories[0]
                        pool[lagged_cov] = np.where(pool[time_name] >= lagged_nums[i],
                                                        pool.groupby([id])[cov].shift(lagged_nums[i]), fill_values)
                    pool[lagged_cov] = pd.Categorical(pool[lagged_cov])
                else:
                    if below_zero_indicator:
                        pool[lagged_cov] = np.array(pool.groupby([id])[cov].shift(lagged_nums[i]))
                    else:
                        fill_values = pool.groupby(id)[cov].transform('first') if baselags else 0
                        pool[lagged_cov] = np.where(pool[time_name] >= lagged_nums[i],
                                                    pool.groupby([id])[cov].shift(lagged_nums[i]), fill_values)

        if len(cov_hist[cov]['cumavg']) > 0:  # create cumavg variable
            pool['_'.join(['cumavg', str(cov)])] = np.array(pool.groupby([id])[cov].expanding().mean())

        lagavg_covs = cov_hist[cov]['lagavg'][0]
        lagavg_nums = cov_hist[cov]['lagavg'][1]
        if len(lagavg_covs) > 0:  # create lagavg variable
            if len(cov_hist[cov]['cumavg']) == 0:  # if cumavg variable has not been created yet, create cumavg variable
                pool['_'.join(['cumavg', str(cov)])] = np.array(pool.groupby([id])[cov].expanding().mean())

            for i, lagavg_cov in enumerate(lagavg_covs):
                if below_zero_indicator:
                    pool[lagavg_cov] = np.array(pool.groupby([id])['_'.join(['cumavg', str(cov)])].shift(lagavg_nums[i]))
                else:
                    fill_values = pool.groupby(id)[cov].transform('first') if baselags else 0
                    pool[lagavg_cov] = np.where(pool[time_name] >= lagavg_nums[i],
                                           pool.groupby([id])['_'.join(['cumavg', str(cov)])].shift(lagavg_nums[i]), fill_values)


def ave_last3(pool, histvar, time_name, t, id):
    """
    This is an example historical function which generates the average of the three most recent values for a specified
    covariate.

    Parameters
    ----------
    pool : DataFrame
        A DataFrame that contains the observed or simulated data up to time t. The historical term at time t in the data
        table is to be updated.

    histvar : Str
        A string that specifies the name of the variable for which the history function is to be applied.

    time_name : Str
        A string specifying the name of the time variable in pool.

    t : Int
         An integer specifying the current time index.

    id : Str
        A string specifying the name of the id variable in the obs_data.

    Returns
    -------
    None : The original input pool has been updated and nothing is returned.

    """
    def avg_func(df, time_name, t, histvar):
        if t < 3:
            avg_values = np.mean((df[(df[time_name] >= 0) & (df[time_name] <= t)][histvar]))
        else:
            avg_values = np.mean((df[(df[time_name] > t - 3) & (df[time_name] <= t)][histvar]))
        return avg_values

    valid_pool = pool.groupby(id).filter(lambda x: max(x[time_name]) >= t)
    pool.loc[pool[time_name] == t, '_'.join(['ave_last3', str(histvar)])] = list(valid_pool.groupby(id).apply(
        avg_func, time_name=time_name, t=t, histvar=histvar))


def update_custom_history(pool, histvars, histories, time_name, t, id):
    """
    This internal function is used to add new columns to the original pool for the user-specified custom historical
    terms.

    Parameters
    ----------
    pool :  DataFrame
        A DataFrame that contains the observed or simulated data up to time t. The historical term at time t in the data
        table is to be updated.

    histvars : List
        A list of strings, each of which specifies the name of the variable for which its custom history function
        is to be applied.

    histories : List
        A list of custom functions, each of which is applied to the variable with the same index in histvars.

    time_name : Str
        A string specifying the name of the time variable in obs_data.

    t : Int
         An integer specifying the current time index.

    id : Str
        A string specifying the name of the id variable in obs_data.

    Returns
    -------
    None : The original input pool has been updated and nothing is returned.

    """
    for i in range(len(histvars)):
        histories[i](pool=pool, histvar=histvars[i], time_name=time_name, t=t, id=id)
