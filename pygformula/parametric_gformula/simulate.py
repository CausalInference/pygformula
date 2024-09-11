import numpy as np
import pandas as pd
import re
import math
import types
from functools import reduce
import operator
from scipy.stats import truncnorm
from .histories import update_precoded_history, update_custom_history
from ..utils.helper import categorical_func
from ..interventions import intervention_func


def binorm_sample(prob):
    return np.random.binomial(n=1, p=prob, size=1)[0]

def norm_sample(mean, rmse):
    return np.random.normal(loc=mean, scale=rmse, size=1)[0]

def truc_sample(mean, rmse, a, b):
    return truncnorm.rvs((a - mean) / rmse, (b - mean) / rmse, loc=mean, scale=rmse)


def simulate(seed, time_points, time_name, id, obs_data, basecovs,
             outcome_type, rmses, bounds, intervention,
             custom_histvars, custom_histories, covpredict_custom, ymodel_predict_custom, ymodel, outcome_fit, outcome_name,
             competing, compevent_name, compevent_model, compevent_fit, compevent_cens, trunc_params,
             visit_names, visit_covs, ts_visit_names, max_visits, time_thresholds, baselags, below_zero_indicator,
             restrictions, yrestrictions, compevent_restrictions, covnames, covtypes, covmodels,
             covariate_fits, cov_hist, sim_trunc):

    """
    This is an internal function to perform Monte Carlo simulation of the parametric g-formula.

    Parameters
    ----------
    seed: Int
        An integer indicating the starting seed for simulations and bootstrapping.

    time_points: Int
        An integer indicating the number of time points to simulate. It is set equal to the maximum number of records
        that obs_data contains for any individual plus 1, if not specified by users.

    time_name: Str
        A string specifying the name of the time variable in obs_data.

    id: Str
        A string specifying the name of the id variable in obs_data.

    covtypes: List
        A list of strings specifying the “type” of each time-varying covariate included in covnames.
        The supported types: "binary", "normal", "categorical", "bounded normal", "zero-inflated normal",
        "truncated normal", "absorbing", "categorical time" "square time" and "custom". The list must be the same length
        as covnames and in the same order.

    obs_data: DataFrame
        A data frame containing the observed data.

    covnames: List
        A list of strings specifying the names of the time-varying covariates in obs_data.

    basecovs: List
        A list of strings specifying the names of baseline covariates in obs_data. These covariates should not be
        included in covnames.

    covmodels: List
        A list of strings, where each string is the model statement of the time-varying covariate. The list
        must be the same length as covnames and in the same order. If a model is not required for a certain covariate,
        it should be set to 'NA' at that index.

    cov_hist: Dict
        A dictionary whose keys are covariate names and values are sub-dictionaries with historical information for
        covariates. Each sub-dictionaty contains keys 'lagged', 'cumavg' and 'lagavg', the corresponding value for the
        key 'lagged' is a two-element list where the first element is a list with all lagged terms, the second element
        is a list with the corresponding lagged numbers. Same for the key 'lagavg'. The corresponding value for the key
        'cumavg' is a list with all cumavg terms.

    cov_fits: List
         A list that contains the fitted model for all time-varying covariates.

    outcome_type: Str
        A string specifying the "type" of outcome. The possible "types" are: "survival", "continuous_eof", and "binary_eof".

    rmses: List
        A list that contains the root mean square errors (rmses) of all the fitted models.

    bounds: List
        A list that contains the bound for all time-varying covariates in the obs_data.

    intervention: List
        List of lists. The k-th list contains the intervention list on k-th treatment name in the intervention.
        The intervention list contains a function implementing a particular intervention on the treatment variable,
        required values for the intervention function and a list of time points in which the intervention
        is applied.

    custom_histvars: List
        A list of strings, each of which specifies the names of the time-varying covariates with user-specified custom histories.

    custom_histories: List
        A list of functions, each function is the user-specified custom history functions for covariates. The list
        should be the same length as custom_histvars and in the same order.

    covpredict_custom: List
        A list, each element could be 'NA' or a user-specified predict function. The non-NA value is set
        for the covariates with custom type. The 'NA' value is set for other covariates. The list must be the
        same length as covnames and in the same order.

    ymodel_predict_custom: Function
        A user-specified predict function for the outcome variable.

    ymodel: Str
        A string specifying the model statement for the outcome variable.

    outcome_fit: Class
        A class object of the fitted model for outcome.

    outcome_name: Str
        A string specifying the name of the outcome variable in obs_data.

    competing: Bool
        A boolean value indicating if there is a competing event in obs_data.

    compevent_name: Str
        A string specifying the name of the competing event variable in obs_data. Only applicable for survival outcomes.

    compevent_model: Str
        A string specifying the model statement for the competing event variable. Only applicable for survival outcomes.

    compevent_fit: Class
        A class object of the fitted model for the competing event.

    compevent_cens: Bool
        A boolean value indicating whether to treat competing events as censoring events.

    trunc_params: List
        A list, each element could be 'NA' or a two-element list. If not 'NA', the first element specifies the truncated
        value and the second element specifies the truncated direction (‘left’ or ‘right’). The non-NA value is set
        for the truncated normal covariates. The 'NA' value is set for other covariates. The list should be the same
        length as covnames and in the same order.

    visit_names: List
        A list, each of which is a string specifying the covariate name of a visit process.

    visit_covs: List
        A list of strings, each of which specifying the name of a covariate whose modeling depends on the visit process.

    ts_visit_names: List
        A list of strings, each of which indicates the number of consecutive missed visits for one covariate before an
        individual is censored.

    max_visits: List
        A list of integers, each integer indicates the maximum number of consecutive missed visits for one covariate that
        has a visit process.

    time_thresholds: List
        A list of integers that splits the time points into different intervals. It is used to create the variable
        "categorical time".

    baselags: Bool
        A boolean value specifying the convention used for lagi and lag_cumavgi terms in the model statements when
        pre-baseline times are not included in obs_data and when the current time index, t, is such that t < i. If this
        argument is set to False, the value of all lagi and lag_cumavgi terms in this context are set to 0 (for
        non-categorical covariates) or the reference level (for categorical covariates). If this argument is set to
        True, the value of lagi and lag_cumavgi terms are set to their values at time 0. The default is False.

    below_zero_indicator: Bool
        A boolean value indicating if the obs_data contains pre-baseline times.

    restrictions: List
        List of lists. Each inner list contains its first entry the covariate name of that its deterministic knowledge
        is known; its second entry is a dictionary whose key is the conditions which should be True when the covariate
        is modeled, the third entry is the value that is set to the covariate during simulation when the conditions
        in the second entry are not True.

    yrestrictions: List
        List of lists. For each inner list, its first entry is a dictionary whose key is the conditions which
        should be True when the outcome is modeled, the second entry is the value that is set to the outcome during
        simulation when the conditions in the first entry are not True.

    compevent_restrictions: List, default is None
        List of lists. For each inner list, its first entry is a dictionary whose key is the conditions which
        should be True when the competing event is modeled, the second entry is the value that is set to the competing
        event during simulation when the conditions in the first entry are not True. Only applicable for survival outcomes.

    sim_trunc: Bool
        A boolean value indicating if the simulated values of normal covariates are truncated by the observed ranges.

    Returns
    -------
    g_result: List
        A list contains the parametric estimate of risks at all the time points for a particular intervention.

    pool: DataFrame
        A simulated data frame under a particular intervention.

    """

    np.random.seed(seed)
    if basecovs:
        column_names = [id] + [time_name] + covnames + basecovs if covnames is not None else [id] + [time_name] + basecovs
    else:
        column_names = [id] + [time_name] + covnames if covnames is not None else [id] + [time_name]
    if ts_visit_names:
        column_names.extend(ts_visit_names)
    pool = obs_data.loc[:, column_names]

    for t in range(0, time_points):
        if t == 0:
            pool = pool[pool[time_name] <= t].copy()
            new_df = pool[pool[time_name] == t]

            intervention_func(new_df=new_df, pool=pool, intervention=intervention, time_name=time_name, t=t)

            pool.loc[pool[time_name] == t] = new_df
            if covnames is not None:
                update_precoded_history(pool, covnames, cov_hist, covtypes, time_name, id, below_zero_indicator,
                                      baselags, ts_visit_names)
                if custom_histvars is not None:
                    update_custom_history(pool, custom_histvars, custom_histories, time_name, t, id)
            new_df = pool[pool[time_name] == t].copy()

            if competing and not compevent_cens:
                prob_D = compevent_fit.predict(new_df)
                new_df['prob_D'] = prob_D

                if compevent_restrictions is not None:
                    for restriction in compevent_restrictions:
                        conditions = restriction[0]
                        masks = []
                        for cond_var, condition in conditions.items():
                            mask = new_df[cond_var].apply(condition)
                            masks.append(mask)
                        comp_restrict_mask = reduce(operator.and_, masks)
                        new_df.loc[~comp_restrict_mask, 'prob_D'] = restriction[1]

                new_df[compevent_name] = new_df['prob_D'].apply(binorm_sample)

            if ymodel_predict_custom is not None:
                pre_y = ymodel_predict_custom(ymodel=ymodel, new_df=new_df, fit=outcome_fit)
            else:
                pre_y = outcome_fit.predict(new_df)

            if outcome_type == 'survival':
                new_df['prob1'] = pre_y

                if yrestrictions is not None:
                    for restriction in yrestrictions:
                        conditions = restriction[0]
                        masks = []
                        for cond_var, condition in conditions.items():
                            mask = new_df[cond_var].apply(condition)
                            masks.append(mask)
                        restrict_mask = reduce(operator.and_, masks)
                        new_df.loc[~restrict_mask, 'prob1'] = restriction[1]

                new_df['prob0'] = 1 - new_df['prob1']
                new_df[outcome_name] = new_df['prob1'].apply(binorm_sample)

            if outcome_type == 'binary_eof':
                new_df['Py'] = 'NA' if t < time_points - 1 else pre_y
            if outcome_type == 'continuous_eof':
                new_df['Ey'] = 'NA' if t < time_points - 1 else pre_y

            if competing and not compevent_cens:
                new_df.loc[new_df[compevent_name] == 1, outcome_name] = 'NA'

            pool = pd.concat([pool[pool[time_name] < t], new_df])
            pool.sort_values([id, time_name], ascending=[True, True], inplace=True)

        else:
            new_df = pool[pool[time_name] == t-1].copy()
            new_df[time_name] = t

            if covtypes is not None:
                if 'categorical time' in covtypes:
                    new_df.loc[new_df[time_name] == t, time_name + '_f'] = new_df[time_name].apply(categorical_func, time_thresholds=time_thresholds)
                if 'square time' in covtypes:
                    new_df.loc[new_df[time_name] == t, 'square_' + time_name] = new_df[time_name] * new_df[time_name]
            pool = pd.concat([pool, new_df])
            pool.sort_values([id, time_name], ascending=[True, True], inplace=True)

            if covnames is not None:
                update_precoded_history(pool, covnames, cov_hist, covtypes, time_name, id, below_zero_indicator,
                                      baselags, ts_visit_names)
                if custom_histvars is not None:
                    update_custom_history(pool, custom_histvars, custom_histories, time_name, t, id)
                new_df = pool[pool[time_name] == t].copy()
                for k, cov in enumerate(covnames):
                    if covmodels[k] != 'NA':
                        if visit_names and cov in visit_names: ### assign values for visit indicator
                            estimated_mean = covariate_fits[cov].predict(new_df)
                            prediction = estimated_mean.apply(binorm_sample)
                            max_visit = max_visits[visit_names.index(cov)]
                            ts_visit_name = ts_visit_names[visit_names.index(cov)]
                            new_df[cov] = np.where(new_df['lag1_{0}'.format(ts_visit_name)] < max_visit, prediction, 1)
                            new_df[ts_visit_name] = np.where(new_df[cov] == 0, new_df[ts_visit_name] + 1, 0)

                        elif covtypes[k] == 'binary':
                            estimated_mean = covariate_fits[cov].predict(new_df)
                            prediction = estimated_mean.apply(binorm_sample)
                            new_df[cov] = prediction

                        elif covtypes[k] == 'normal':
                            estimated_mean = covariate_fits[cov].predict(new_df)
                            prediction = estimated_mean.apply(norm_sample, rmse=rmses[cov])
                            if sim_trunc:
                                prediction = np.where(prediction < bounds[cov][0], bounds[cov][0], prediction)
                                prediction = np.where(prediction > bounds[cov][1], bounds[cov][1], prediction)
                            new_df[cov] = prediction

                        elif covtypes[k] == 'categorical':
                            predict_probs = covariate_fits[cov].predict(new_df)
                            predict_index = np.asarray(predict_probs).argmax(1)
                            prediction = list(map(lambda x: pd.Categorical(obs_data[cov]).categories[x], predict_index))
                            new_df[cov] = prediction

                        elif covtypes[k] == 'bounded normal':
                            estimated_mean = covariate_fits[cov].predict(new_df)
                            prediction = estimated_mean.apply(norm_sample, rmse=rmses[cov])
                            prediction = prediction.apply(lambda x: x * (bounds[cov][1] - bounds[cov][0]) + bounds[cov][0])
                            prediction = np.where(prediction < bounds[cov][0], bounds[cov][0], prediction)
                            prediction = np.where(prediction > bounds[cov][1], bounds[cov][1], prediction)
                            new_df[cov] = prediction

                        elif covtypes[k] == 'zero-inflated normal':
                            estimated_indicator_mean = covariate_fits[cov][0].predict(new_df)
                            indicator = estimated_indicator_mean.apply(binorm_sample)
                            estimated_mean = covariate_fits[cov][1].predict(new_df)
                            prediction = estimated_mean.apply(norm_sample, rmse=rmses[cov])
                            nonzero_predict = prediction.apply(lambda x: math.exp(x))
                            prediction = indicator * nonzero_predict
                            prediction = np.where((prediction < bounds[cov][0]) & (indicator == 1), bounds[cov][0], prediction)
                            prediction = np.where((prediction > bounds[cov][1]) & (indicator == 1), bounds[cov][1], prediction)
                            new_df[cov] = prediction

                        elif covtypes[k] == 'truncated normal':
                            fit_coefficients = covariate_fits[cov]
                            _, covmodel = re.split('~', covmodels[k].replace(' ', ''))
                            var_names = re.split('\+', covmodel)
                            new_data = np.concatenate((np.ones((new_df.shape[0], 1)), new_df[var_names].to_numpy()), axis=1)
                            estimated_mean = np.dot(new_data, fit_coefficients['x'][:-1])

                            if trunc_params[k][1] == 'left':
                                trunc_bounds = [trunc_params[k][0], float('inf')]
                            else:
                                trunc_bounds = [-float('inf'), trunc_params[k][0]]

                            prediction = pd.Series(estimated_mean).apply(truc_sample, rmse=rmses[cov], a=trunc_bounds[0],
                                                                     b=trunc_bounds[1])
                            prediction = np.where(prediction < bounds[cov][0], bounds[cov][0], prediction)
                            prediction = np.where(prediction > bounds[cov][1], bounds[cov][1], prediction)
                            new_df[cov] = prediction

                        elif covtypes[k] == 'absorbing':
                            predict_prob = covariate_fits[cov].predict(new_df)
                            prediction = predict_prob.apply(binorm_sample)
                            prediction = np.where(pool.loc[pool[time_name] == t - 1, cov] == 0, prediction, 1)
                            new_df[cov] = prediction

                        elif covtypes[k] == 'custom':
                            pred_func = covpredict_custom[k]
                            prediction = pred_func(covmodel=covmodels[k], new_df=new_df, fit=covariate_fits[cov])
                            new_df[cov] = prediction

                        if visit_covs and cov in visit_covs: ### assign visited covariate the model output value or its lagged value based on visit indicator
                            visit_name = visit_names[visit_covs.index(cov)]
                            new_df[cov] = np.where(new_df[visit_name] == 0, new_df['lag1_{0}'.format(cov)], new_df[cov])

                    if restrictions is not None:
                        restrictcovs = [restrictions[i][0] for i in range(len(restrictions))]
                        if cov in restrictcovs:
                            index = restrictcovs.index(cov)
                            restriction = restrictions[index]
                            conditions = restriction[1]
                            masks = []
                            for cond_var, condition in conditions.items():
                                mask = new_df[cond_var].apply(condition)
                                masks.append(mask)
                            restrict_mask = reduce(operator.and_, masks)
                            if isinstance(restriction[2], types.FunctionType):
                                assigned_values = restriction[2](new_df=new_df, pool=pool, time_name=time_name, t=t)
                                new_df.loc[~restrict_mask, cov] = assigned_values
                            else:
                                new_df.loc[~restrict_mask, cov] = restriction[2]

                    pool.loc[pool[time_name] == t] = new_df
                    if len(cov_hist[cov]['cumavg']) > 0:
                        update_precoded_history(pool, covnames, cov_hist, covtypes, time_name, id, below_zero_indicator, baselags, ts_visit_names)
                    if custom_histvars is not None and cov in custom_histvars:
                        update_custom_history(pool, custom_histvars, custom_histories, time_name, t, id)
                    new_df = pool[pool[time_name] == t].copy()

            intervention_func(new_df=new_df, pool=pool, intervention=intervention, time_name=time_name, t=t)

            pool.loc[pool[time_name] == t] = new_df
            if covnames is not None:
                update_precoded_history(pool, covnames, cov_hist, covtypes, time_name, id, below_zero_indicator,
                                      baselags, ts_visit_names)
                if custom_histvars is not None:
                    update_custom_history(pool, custom_histvars, custom_histories, time_name, t, id)
            new_df = pool[pool[time_name] == t].copy()

            if competing and not compevent_cens:
                params_D = re.split('[~|\+]', compevent_model.replace(' ', ''))
                prob_D = compevent_fit.predict(new_df[params_D])
                new_df['prob_D'] = prob_D

                if compevent_restrictions is not None:
                    for restriction in compevent_restrictions:
                        conditions = restriction[0]
                        masks = []
                        for cond_var, condition in conditions.items():
                            mask = new_df[cond_var].apply(condition)
                            masks.append(mask)
                        comp_restrict_mask = reduce(operator.and_, masks)
                        new_df.loc[~comp_restrict_mask, 'prob_D'] = restriction[1]

                new_df[compevent_name] = new_df['prob_D'].apply(binorm_sample)

            if ymodel_predict_custom is not None:
                pre_y = ymodel_predict_custom(ymodel=ymodel, new_df=new_df, fit=outcome_fit)
            else:
                pre_y = outcome_fit.predict(new_df)

            if outcome_type == 'survival':
                new_df['prob1'] = pre_y

                if yrestrictions is not None:
                    for restriction in yrestrictions:
                        conditions = restriction[0]
                        masks = []
                        for cond_var, condition in conditions.items():
                            mask = new_df[cond_var].apply(condition)
                            masks.append(mask)
                        restrict_mask = reduce(operator.and_, masks)
                        new_df.loc[~restrict_mask, 'prob1'] = restriction[1]

                new_df['prob0'] = 1 - new_df['prob1']
                new_df[outcome_name] = new_df['prob1'].apply(binorm_sample)

            if outcome_type == 'binary_eof':
                new_df['Py'] = 'NA' if t < time_points - 1 else pre_y
            if outcome_type == 'continuous_eof':
                new_df['Ey'] = 'NA' if t < time_points - 1 else pre_y

            if competing and not compevent_cens:
                new_df[outcome_name] = 'NA'

            pool.loc[pool[time_name] == t] = new_df

    pool = pool[pool[time_name] >= 0]

    if outcome_type == 'survival':
        if competing and not compevent_cens:
            pool['cumprob0'] = pool.groupby([id])['prob0'].cumprod()
            pool['prob_D0'] = 1 - pool['prob_D']
            pool['cumprob_D0'] = pool.groupby([id])['prob_D0'].cumprod()
            pool['prodp1'] = np.where(pool[time_name] > 0,pool.groupby([id])['cumprob0'].shift(1) * pool['cumprob_D0'].shift(1)
                                      * pool['prob1'] * (1 - pool['prob_D']),
                                      pool['prob1'] * (1 - pool['prob_D']))
            pool['risk'] = pool.groupby([id])['prodp1'].cumsum()
            pool['survival'] = 1 - pool['risk']
            g_result = pool.groupby(time_name, group_keys=False)['risk'].mean().tolist()
        else:
            pool['cumprod0'] = pool.groupby([id])['prob0'].cumprod()
            pool['prodp1'] = np.where(pool[time_name] > 0, pool.groupby([id])['cumprod0'].shift(1) * pool['prob1'], pool['prob1'])
            pool['risk'] = pool.groupby([id])['prodp1'].cumsum()
            pool['survival'] = 1 - pool['risk']
            g_result = pool.groupby(time_name, group_keys=False)['risk'].mean().tolist()

    if outcome_type == 'continuous_eof':
        g_result = pool.loc[pool[time_name] == time_points - 1]['Ey'].mean()

    if outcome_type == 'binary_eof':
        g_result = pool.loc[pool[time_name] == time_points - 1]['Py'].mean()

    return {'g_result': g_result, 'pool': pool}

