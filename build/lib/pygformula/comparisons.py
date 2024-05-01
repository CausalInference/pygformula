import numpy as np


def comparison_calculate(obs_data, time_name, time_points, id, covnames, covtypes, outcome_name, outcome_type,
                         nc_pool, nc_risk, censor, censor_name, censor_fit, ipw_cutoff_quantile, ipw_cutoff_value,
                         competing=None, compevent_name=None, compevent_cens=False, compevent_fit=None):
    """
    This is an internal function to calculate the mean observed values of covariates at each time point, as well as mean
    observed risk.

    Parameters
    ----------
    obs_data: DataFrame
        A data frame containing the observed data.

    time_name: Str
        A string specifying the name of the time variable in obs_data.

    time_points: Int
        An integer indicating the number of time points to simulate. It is set equal to the maximum number of records
        that obs_data contains for any individual plus 1, if not specified by users.

    id: Str
        A string specifying the name of the id variable in obs_data.

    covnames: List
        A list of strings specifying the names of the time-varying covariates in obs_data.

    covtypes: List
        A list of strings specifying the “type” of each time-varying covariate included in covnames.
        The supported types: "binary", "normal", "categorical", "bounded normal", "zero-inflated normal",
        "truncated normal", "absorbing", "categorical time" "square time" and "custom". The list must be the same length
        as covnames and in the same order.

    outcome_name: Str
        A string specifying the name of the outcome variable in obs_data.

    outcome_type: Str
        A string specifying the "type" of outcome. The possible "types" are: "survival", "continuous_eof", and "binary_eof".\

    nc_pool: DataFrame
        A dataframe of the simulated data under natural course.

    nc_risk: List
        A list contains the parametric risk of all the time points for natural course.

    censor: Bool
        A boolean value indicating the if there is a censoring event.

    censor_name: Str
        A string specifying the name of the censoring variable in obs_data. Only applicable when using inverse
        probability weights to estimate the natural course means / risk from the observed data.

    censor_fit: Class
        A class object of the fitted model for the censoring event.

    ipw_cutoff_quantile: Float
        Percentile value for truncation of the inverse probability weights

    ipw_cutoff_value: Float
        Absolute value for truncation of the inverse probability weights.

    competing: Bool
        A boolean value indicating the if there is a competing event.

    compevent_name: Str
        A string specifying the name of the competing event variable in obs_data. Only applicable for survival outcomes.

    compevent_cens: Bool
        A boolean value indicating whether to treat competing events as censoring events.

    compevent_fit: Class
        A class object of the fitted model for the competing event.

    Returns
    -------
    obs_means: Dict
        A dictionary, where the key is the covariate / risk name and the value is its observational mean at all the time points.

    est_means: Dict
        A dictionary, where the key is the covariate / risk name and the value is its parametric mean at all the time points.

    obs_res: Float
        A value of the observational risk / mean at final time point.

    IP_weights: List
        A list contains the inverse probability weights from the censor model.

    """
    if censor:
        # for non-parametric cov means and risks
        censor_pre = censor_fit.predict(obs_data)
        censor_p0_inv = 1 / (1 - censor_pre)
        obs_data['censor_p0_inv'] = censor_p0_inv
        censor_inv_cum = obs_data.groupby([id])['censor_p0_inv'].cumprod()
        obs_data['censor_inv_cum'] = censor_inv_cum
        w_censor = censor_inv_cum * (1 - obs_data[censor_name])
        if outcome_type == 'survival' and compevent_cens:
            comprisk_p0_inv = 1 / (1 - compevent_fit.predict(obs_data))
            obs_data['comprisk_p0_inv'] = comprisk_p0_inv
            comprisk_inv_cum = obs_data.groupby([id])['comprisk_p0_inv'].cumprod()
            w_comp = np.where((obs_data[compevent_name].isna()) | (obs_data[compevent_name] == 1), 0, comprisk_inv_cum)
            w = w_comp * w_censor
        else:
            w = w_censor
        obs_data['IP_weight'] = w

        if ipw_cutoff_quantile:
            quantile_w = np.percentile(list(w_censor), ipw_cutoff_quantile * 100)
            obs_data.loc[obs_data['IP_weight'] > quantile_w, 'IP_weight'] = quantile_w
        if ipw_cutoff_value:
            obs_data.loc[obs_data['IP_weight'] > ipw_cutoff_value, 'IP_weight'] = ipw_cutoff_value

        obs_data['IP_weight_cov'] = np.where(obs_data[time_name] > 0, obs_data['IP_weight'].shift(1), 1)

        obs_means = {}
        if covnames is not None:
            for k, covname in enumerate(covnames):
                if covtypes[k] == 'categorical':
                    all_levels = np.unique(obs_data[covname])
                    all_levels_obs_prob = []
                    for level in all_levels:
                        obs_level_prob = obs_data[obs_data[covname].notna()].groupby([time_name]).apply(lambda g:
                                        (((g[covname] == level) * g['IP_weight_cov']).mean()) /g['IP_weight_cov'].mean()).tolist()[:time_points]
                        all_levels_obs_prob.append(obs_level_prob)
                else:
                    cov_mean = obs_data[obs_data[covname].notna()].groupby(time_name).apply(lambda g: (g['IP_weight_cov'] * g[covname]).mean() / g['IP_weight_cov'].mean()).tolist()[:time_points]
                    obs_means[covname] = cov_mean

        if outcome_type == 'binary_eof' or outcome_type == 'continuous_eof':
            obs_data_last_record =  obs_data.loc[obs_data[outcome_name].notna()]
            obs_mean_Ey = (obs_data_last_record[outcome_name] * obs_data_last_record['IP_weight']).mean() / obs_data_last_record['IP_weight'].mean()

        if outcome_type == 'survival':
            if competing and not compevent_cens:
                w_elimD = obs_data['IP_weight'] * (1 - obs_data[compevent_name])
                obs_data['w_elimD'] = w_elimD
                h_k = obs_data[obs_data[outcome_name].notna()].groupby(time_name).apply(
                    lambda g: (g['w_elimD'] * g[outcome_name]).mean() / g['w_elimD'].mean())
                h_k2 = obs_data[obs_data[compevent_name].notna()].groupby(time_name).apply(
                    lambda g: (g['IP_weight'] * g[compevent_name]).mean() / g['IP_weight'].mean())
                risks = np.array([list(h_k)[k] * (1 - list(h_k2)[k]) if k == 0 else list(h_k)[k]
                                  * (1 - list(h_k2)[k]) * list((1 - h_k).cumprod())[k - 1] * list((1 - h_k2).cumprod())[k - 1]
                                  for k in range(time_points)]).cumsum().tolist()[:time_points]
                obs_means['risk'] = risks
                obs_risk = risks[-1]
            else:
                weight_outcome_mean = obs_data[obs_data[outcome_name].notna()].groupby(time_name).apply(
                    lambda g: (g['IP_weight'] * g[outcome_name]).mean() / g['IP_weight'].mean())
                weight_p0_mean = 1 - weight_outcome_mean
                risks = np.array([weight_outcome_mean.tolist()[k] if k == 0 else weight_outcome_mean.tolist()[k] *
                                 weight_p0_mean.cumprod().tolist()[k - 1] for k in
                                 range(time_points)]).cumsum().tolist()[:time_points]
                obs_means['risk'] = risks
                obs_risk = risks[-1]

        if outcome_type == 'survival':
            # for parametric cov means and risks
            if competing and not compevent_cens:
                nc_pool['p0_cum'] = nc_pool.groupby(id)['prob0'].cumprod()
                nc_pool['pd_0'] = 1 - nc_pool['prob_D']
                nc_pool['pd_0_cum'] = nc_pool.groupby(id)['pd_0'].cumprod()
                nc_pool['w_cov'] = np.where(nc_pool[time_name] > 0,
                                            nc_pool['p0_cum'].shift(1) * nc_pool['pd_0_cum'].shift(1), 1)
            else:
                nc_pool['p0_cum'] = nc_pool.groupby([id])['prob0'].cumprod()
                nc_pool['w_cov'] = np.where(nc_pool[time_name] > 0, nc_pool['p0_cum'].shift(1), 1)
        else:
            nc_pool['w_cov'] = 1

        est_means = {}
        if covnames is not None:
            for k, covname in enumerate(covnames):
                if covtypes[k] == 'categorical':
                    all_levels = np.unique(obs_data[covname])
                    all_levels_est_prob_mean = []
                    for level in all_levels:
                        est_level_prob = nc_pool[nc_pool[covname].notna()].groupby([time_name]).apply(
                            lambda g: ((g[covname] == level) * g['w_cov']).mean() / g['w_cov'].mean()).tolist()[:time_points]
                        all_levels_est_prob_mean.append(est_level_prob)
                    est_means[covname] = all_levels_est_prob_mean
                else:
                    cov_mean = nc_pool[nc_pool[covname].notna()].groupby(time_name).apply(
                        lambda g: (g['w_cov'] * g[covname]).mean() / g['w_cov'].mean()).tolist()[:time_points]
                    est_means[covname] = cov_mean
        if outcome_type == 'survival':
            est_means['risk'] = nc_risk

    else:
        # for non-parametric cov means and risks
        obs_means = {}
        if covnames is not None:
            for k, covname in enumerate(covnames):
                if covtypes[k] == 'categorical':
                    all_levels = np.unique(obs_data[covname])
                    all_levels_obs_prob_mean = []
                    for level in all_levels:
                        obs_level_prob = obs_data[obs_data[covname].notna()].groupby([time_name]).apply(lambda g: ((g[covname] == level)).mean()).tolist()[:time_points]
                        all_levels_obs_prob_mean.append(obs_level_prob)
                    obs_means[covname] = all_levels_obs_prob_mean
                else:
                    obs_mean = obs_data[obs_data[covname].notna()].groupby([time_name])[covname].mean().tolist()[:time_points]
                    obs_means[covname] = obs_mean

        if outcome_type == 'binary_eof' or outcome_type == 'continuous_eof':
            obs_mean_Ey = obs_data.loc[obs_data[time_name] == time_points - 1][outcome_name].mean()

        if outcome_type == 'survival':
            if competing and not compevent_cens:
                p1_mean = obs_data[obs_data[outcome_name].notna()].groupby(time_name)[outcome_name].mean()
                pd_mean = obs_data[obs_data[compevent_name].notna()].groupby(time_name)[compevent_name].mean()
                comrisks = np.array(
                    [list(p1_mean)[k] * (1 - list(pd_mean)[k]) if k == 0 else list(p1_mean)[k] * (1 - list(pd_mean)[k]) *
                     list((1 - p1_mean).cumprod())[k - 1] * list((1 - pd_mean).cumprod())[k - 1]
                     for k in range(time_points)]).cumsum().tolist()[:time_points]
                obs_means['risk'] = comrisks
                obs_risk = comrisks[-1]
            else:
                p1_mean_obs = obs_data[obs_data[outcome_name].notna()].groupby(time_name)[outcome_name].mean()
                p0_mean_obs = 1 - p1_mean_obs
                risks = np.array(
                    [p1_mean_obs.tolist()[k] if k == 0 else p1_mean_obs.tolist()[k] * p0_mean_obs.cumprod().tolist()[k - 1]
                     for k in range(time_points)]).cumsum().tolist()[:time_points]
                obs_means['risk'] = risks
                obs_risk = risks[-1]

        if outcome_type == 'survival':
            # for parametric cov means and risks
            if competing and not compevent_cens:
                nc_pool['p0_cum'] = nc_pool.groupby(id)['prob0'].cumprod()
                nc_pool['pd_0'] = 1 - nc_pool['prob_D']
                nc_pool['pd_0_cum'] = nc_pool.groupby(id)['pd_0'].cumprod()
                nc_pool['w_cov'] = np.where(nc_pool[time_name] > 0,
                                            nc_pool['p0_cum'].shift(1) * nc_pool['pd_0_cum'].shift(1), 1)
            else:
                nc_pool['p0_cum'] = nc_pool.groupby([id])['prob0'].cumprod()
                nc_pool['w_cov'] = np.where(nc_pool[time_name] > 0, nc_pool['p0_cum'].shift(1), 1)
        else:
            nc_pool['w_cov'] = 1

        est_means = {}
        if covnames is not None:
            for k, covname in enumerate(covnames):
                if covtypes[k] == 'categorical':
                    all_levels = np.unique(obs_data[covname])
                    all_levels_est_prob_mean = []
                    for level in all_levels:
                        est_level_prob = nc_pool[nc_pool[covname].notna()].groupby([time_name]).apply(
                            lambda g: ((g[covname] == level) * g['w_cov']).mean() / g['w_cov'].mean()).tolist()[:time_points]
                        all_levels_est_prob_mean.append(est_level_prob)
                    est_means[covname] = all_levels_est_prob_mean
                else:
                    est_mean = nc_pool[nc_pool[covname].notna()].groupby(time_name).apply(lambda g:
                                                                              (g['w_cov'] * g[covname]).mean()
                                                                             / g['w_cov'].mean()).tolist()[:time_points]
                    est_means[covname] = est_mean
        if outcome_type == 'survival':
            est_means['risk'] = nc_risk

    obs_res = obs_risk if outcome_type == 'survival' else obs_mean_Ey
    IP_weights = obs_data['IP_weight'].tolist() if censor else None

    return obs_means, est_means, obs_res, IP_weights

