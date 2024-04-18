import re
import numpy as np


def get_cov_hist_info(covnames, covmodels, covtypes, ymodel, compevent_model=None, censor_model=None,
                      visit_covs=None, ts_visit_names=None):
    """
    This is an internal function to get the lagged term and its number indicator, cumavg term, and lagavg term and its number
    indicator for each covariate from user-specified models.

    Parameters
    ----------
    covnames : List
        A list of strings specifying the names of the time-varying covariates in obs_data.

    covmodels : List
        A list of strings, where each string is the model statement of the time-varying covariate. The list must be the
        same length as covnames and in the same order. If a model is not required for a certain covariate, it should be
        set to 'NA' at that index.

    covtypes : List
        A list of strings specifying the “type” of each time-varying covariate included in covnames. The supported types:
        "binary", "normal", "categorical", "bounded normal", "zero-inflated normal", "truncated normal", "absorbing",
        "categorical time", "square time" and "custom". The list must be the same length as covnames and in the same order.

    ymodel : Str
        A string specifying the model statement for the outcome variable.

    compevent_model : Str, (default=None)
        A string specifying model statement for the competing event variable. Only applicable for survival outcomes.

    censor_model : Str, (default=None)
        A string specifying the model statement for the censoring variable. Only applicable when using inverse
        probability weights to estimate the natural course means / risk from the observed data.

    visit_covs : List, (default=None)
        A list of strings, each of which specifies the name of a covariate whose modeling depends on the visit process.

    ts_visit_names : List, (default=None)
        A list of strings, each of which indicates the number of consecutive missed visits for one covariate before an
        individual is censored. The list has the same length as visit_covs.

    Returns
    -------
    cov_hist_infos : Dict
        A dictionary whose keys are covariate names and values are sub-dictionaries with historical information for
        covariates. Each sub-dictionaty contains keys 'lagged', 'cumavg' and 'lagavg', the corresponding value for the
        key 'lagged' is a two-element list where the first element is a list with all lagged terms, the second element
        is a list with the corresponding lagged numbers. Same for the key 'lagavg'. The corresponding value for the key
        'cumavg' is a list with all cumavg terms.

    """

    all_variables = []
    for model in covmodels:
        all_variables.extend(re.split('[~|+]', model.replace(' ', '')))
    all_variables.extend(re.split('[~|+]', ymodel.replace(' ', '')))

    if compevent_model is not None:
        all_variables.extend(re.split('[~|+]', compevent_model.replace(' ', '')))
    if censor_model is not None:
        all_variables.extend(re.split('[~|+]', censor_model.replace(' ', '')))

    if ts_visit_names:
        covnames = covnames + ts_visit_names

    cov_hist_infos = {}
    for k, cov in enumerate(covnames):
        cov_list = np.unique([str_cov for str_cov in all_variables if cov in str_cov])
        if k < len(covtypes):
            if covtypes[k] == 'absorbing':
                cov_list = np.append(cov_list, 'lag1_{0}'.format(cov))
        if visit_covs and cov in visit_covs:
            cov_list = np.append(cov_list, 'lag1_{0}'.format(cov))
        if ts_visit_names and cov in ts_visit_names:
            cov_list = np.append(cov_list, 'lag1_{0}'.format(cov))

        cov_hist = {}
        lagavg_variables, cumavg_variables, lagged_variables = [], [], []
        lagged_numbers, lagavg_numbers = [], []
        for item in cov_list:
            if 'lag' in item and 'lag_cumavg' not in item:
                pattern = re.compile(r'lag\d+_{0}'.format(cov))
                lag_names = pattern.findall(item)
                for lag_name in lag_names:
                    lagged_variables.append(lag_name)
                    lagged_numbers.append(int(lag_name.split('_')[0].split('lag')[1]))

            if 'cumavg' in item and 'lag_cumavg' not in item:
                if covtypes[k] == 'categorical' or covtypes[k] == 'categorical time':
                    raise ValueError('Cannot apply cumulative average function to categorical covariates.')
                pattern = re.compile(r'cumavg_{0}'.format(cov))
                cumavg_names = pattern.findall(item)
                for cumavg_name in cumavg_names:
                    cumavg_variables.append(cumavg_name)

            if 'lag_cumavg' in item:
                if covtypes[k] == 'categorical' or covtypes[k] == 'categorical time':
                    raise ValueError('Cannot apply lagged cumulative average function to categorical covariates.')
                pattern = re.compile(r'lag_cumavg\d+_{0}'.format(cov))
                lagavg_names = pattern.findall(item)
                for lagavg_name in lagavg_names:
                    lagavg_variables.append(lagavg_name)
                    lagavg_numbers.append(int(lagavg_name.split('_')[1].split('cumavg')[1]))

        cov_hist['lagged'] = [lagged_variables, lagged_numbers]
        cov_hist['cumavg'] = cumavg_variables
        cov_hist['lagavg'] = [lagavg_variables, lagavg_numbers]
        cov_hist_infos[cov] = cov_hist

    return cov_hist_infos


def visit_func(df, time_name, visit_name, ts_visit_name):
    """
    An internal function assists the implementation of a visit process, it creates a new column named ts_visit_name.

    Parameters
    ----------
    df : DataFrame
        A pandas DataFrame of the input obs_data.

    time_name : Str
        A string specifying the name of the time variable in obs_data.

    visit_name : Str
        A string specifying the covariate name of a visit process.

    ts_visit_name : Str
        A string indicating the number of consecutive missed visits before an individual is censored.

    Returns
    -------
    df : DataFrame
        A pandas DataFrame with a new column ts_visit_name created.

    """

    df.loc[df[time_name] == 0, ts_visit_name] = 0
    tp_visits = 0
    for t in range(1, max(df[time_name]) + 1):
        if df.loc[df[time_name] == t, visit_name].values[0] == 1:
            df.loc[df[time_name] == t, ts_visit_name] = 0
        else:
            if df.loc[df[time_name] == t - 1, visit_name].values[0] == 1:  # restart the count with new visit
                df.loc[df[time_name] == t, ts_visit_name] = 1
                tp_visits = 0
            else:  # continue to count the missed visit number
                tp_visits += 1
                df.loc[df[time_name] == t, ts_visit_name] = 1 + tp_visits
    return df


def categorical_func(t, time_thresholds):
    for i in range(len(time_thresholds)):
        if t <= time_thresholds[i]:
            categorical_t = i
            break
        else:
            categorical_t = i + 1
    return categorical_t


def hr_data_helper(df, outcome_name):
    for i, row in df.iterrows():
        if row[outcome_name] == 1:
            return row
    return row


def hr_comp_data_helper(df, outcome_name, compevent_name):
    for i, row in df.iterrows():
        if row[compevent_name] == 1:
            return row
        elif row[outcome_name] == 1:
            return row
    return row