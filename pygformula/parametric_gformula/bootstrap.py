import numpy as np
import pandas as pd
from lifelines import CoxPHFitter
from .histories import update_precoded_history, update_custom_history
from .simulate import simulate
from .fit import fit_covariate_model, fit_outcome_model, fit_compevent_model
from ..utils.helper import hr_data_helper, hr_comp_data_helper


def Bootstrap(obs_data, boot_id, boot_seeds, interventions, int_descripts, intvars, int_times, covnames,
              basecovs, cov_hist, time_points, n_simul, time_name, id_name, custom_histvars, custom_histories,
              covmodels, hazardratio, intcomp, covtypes, covfits_custom, covpredict_custom,
              outcome_model, outcome_type, outcome_name, competing, compevent_name, compevent_model, compevent_cens,
              boot_diag, trunc_params, visit_names, visit_covs, ts_visit_names, max_visits, time_thresholds,
              below_zero_indicator, baselags, restrictions, yrestrictions, compevent_restrictions):
    """
    This is an internal function to get the results of parametric g-formula for each bootstrap sample.

    Parameters
    ----------
    obs_data: DataFrame
        A data frame containing the observed data.

    boot_id: Int
        An interger indicating the id of the bootstrap sample.

    boot_seeds: List
        A list that stores the random seeds of all bootstrap samples.

    interventions: List
        A list whose elements are lists of vectors. Each list in interventions specifies a unique intervention on the
        relevant variable(s) in intvars. Each vector contains a function implementing a particular intervention on a
        single variable, and required values for specific treatment strategy.

    int_descripts: List
        A list of strings, each of which describes a user-specified intervention.

    intvars: List
        A list, each element is a list of strings. The kth element in intvars specifies the name(s) of the variable(s) to
        be intervened on under the kth intervention in interventions.

    int_times: List
        A list, each element is a list. The kth list in int_times corresponds to the kth intervention in interventions.
        Each inner list specifies the time points in which the relevant intervention is applied on the corresponding
        variable in intvars. By default, this argument is set so that all interventions are applied in all the time points.

    covnames: List
        A list of strings specifying the names of the time-varying covariates in obs_data.

    basecovs: List
        A vector of strings specifying the names of baseline covariates in obs_data. These covariates are not simulated
        using a model but keep the same value at all time steps. These covariates should not be included in covnames.

    cov_hist: Dict
        A dictionary whose keys are covariate names and values are sub-dictionaries with historical information for
        covariates. Each sub-dictionaty contains keys 'lagged', 'cumavg' and 'lagavg', the corresponding value for the
        key 'lagged' is a two-element list where the first element is a list with all lagged terms, the second element
        is a list with the corresponding lagged numbers. Same for the key 'lagavg'. The corresponding value for the key
        'cumavg' is a list with all cumavg terms.

    time_points: Int
        An integer indicating the number of time points to simulate.

    n_simul: Int
        An integer indicating the number of subjects for whom to simulate data.

    time_name: Str
        A string specifying the name of the time variable in obs_data.

    id_name: Str
        A string specifying the name of the id variable in obs_data.

    custom_histvars: List
        A list of strings, each of which specifies the names of the time-varying covariates with user-specified custom histories.

    custom_histories: List
        A list of function, each function is the user-specified custom history functions for covariates. The list must
        be the same length as custom_histvars and in the same order.

    covmodels: List
        A list of strings, where each string is the model statement of the time-varying covariate. The list must be the
        same length as covnames and in the same order. If a model is not required for a certain covariate, it should be
        set to 'NA' at that index.

    hazardratio: Bool
        A boolean value indicating whether to calculate the hazard ratio of the two compared interventions.

    intcomp: List
        A list of two numbers indicating a pair of interventions to be compared by a hazard ratio.

    covtypes: List
        A list of strings specifying the “type” of each time-varying covariate included in covnames. The supported types:
        "binary", "normal", "categorical", "bounded normal", "zero-inflated normal", "truncated normal", "absorbing",
        "categorical time", "square time" and "custom". The list must be the same length as covnames and in the same order.

    covfits_custom: List
        A list, at the index where the covtype is set to "custom", the element is a user-specified fit function, otherwise
        it should be set to 'NA'. The list must be the same length as covnames and in the same order.

    covpredict_custom: List
        A list, at the index where the covtype is set to "custom", the element is a user-specified predict function,
        otherwise it should be set to 'NA'.

    outcome_model: Str
        A string specifying the model statement for the outcome variable.

    outcome_type: Str
        A string specifying the "type" of outcome. The possible "types" are: "survival", "continuous_eof", and "binary_eof".

    outcome_name: Str
        A string specifying the name of the outcome variable in obs_data.

    competing: Bool
        A boolean value indicating if there is a competing event in obs_data.

    compevent_name: Str
        A string specifying the name of the competing event variable in obs_data. Only applicable for survival outcomes.

    compevent_model: Str
        A string specifying the model statement for the competing event variable. Only applicable for survival outcomes.

    compevent_cens: Bool
        A boolean value indicating whether to treat competing events as censoring events.

    boot_diag: Bool
        A boolean value indicating whether to return the parametric g-formula estimates as well as the coefficients,
        standard errors, and variance-covariance matrices of the parameters of the fitted models in the bootstrap samples.

    trunc_params: List
        A list, at the index where the covtype is set to "truncated normal", the element contains two elements.
        The first element specifies the truncated value and the second element specifies the truncated direction
        (‘left’ or ‘right’). The values at remaining indexes are set to 'NA'. The list must be the same length as
        covnames and in the same order.

    visit_names: List
        A list, each of which is a string specifying the covariate name of a visit process.

    visit_covs: List
        A list of strings, each of which specifies the name of a covariate whose modeling depends on the visit process.

    ts_visit_names: List
        A list of strings, each of which indicates the number of consecutive missed visits for one covariate before an
        individual is censored.

    max_visits: List
        A list of integars, each integar indicates the maximum number of consecutive missed visits for one covariate that
        has a visit process.

    time_thresholds: List
        A list of integars that splits the time points into different intervals. It is used to create the time variable
        of "categorical time".

    below_zero_indicator: Bool
        A boolean value indicating if the obs_data contains pre-baseline times.

    baselags: Bool
        A boolean value specifying the convention used for lagi and lag_cumavgi terms in the model statements when
        pre-baseline times are not included in obs_data and when the current time index, t, is such that t < i. If this
        argument is set to False, the value of all lagi and lag_cumavgi terms in this context are set to 0 (for
        non-categorical covariates) or the reference level (for categorical covariates). If this argument is set to
        True, the value of lagi and lag_cumavgi terms are set to their values at time 0. The default is False.

    restrictions: List
        A list with lists, each inner list contains its first entry the covariate name of that its deterministic knowledge
        is known; its second entry is a dictionary whose key is the conditions which should be True when the covariate
        is modeled, the third entry is the value that is set to the covariate during simulation when the conditions
        in the second entry are not True.

    yrestrictions: List
        A list with lists, for each inner list, its first entry is a dictionary whose key is the conditions which
        should be True when the outcome is modeled, the second entry is the value that is set to the outcome during
        simulation when the conditions in the first entry are not True.

    compevent_restrictions: List
        A list with lists, for each inner list, its first entry is a dictionary whose key is the conditions which
        should be True when the competing event is modeled, the second entry is the value that is set to the competing
        event during simulation when the conditions in the first entry are not True. Only applicable for survival outcomes.

    Returns
    -------
    boot_results_dict: Dict
        A dictionary contains the 'boot_results', 'bootcoeffs', 'bootstderrs', 'bootvcovs' and 'boot_hr' for a bootstrap sample.

    """

    np.random.seed(boot_seeds[boot_id])

    data_list = dict(list(obs_data.groupby(id_name, group_keys=False)))
    ids = np.unique(obs_data[id_name])
    new_ids = np.random.choice(ids, len(ids), replace=True)

    new_df = []
    for index, new_id in enumerate(new_ids):
        new_id_df = data_list[new_id].copy()
        new_id_df[id_name] = index
        new_df.append(new_id_df)
    resample_data = pd.concat(new_df, ignore_index=True)

    update_precoded_history(pool=resample_data, covnames=covnames, cov_hist=cov_hist, covtypes=covtypes,
                            time_name=time_name, id_name=id_name, below_zero_indicator=below_zero_indicator,
                            baselags=baselags, ts_visit_names = ts_visit_names)
    if custom_histvars is not None:
        for t in range(time_points):
            update_custom_history(resample_data, custom_histvars, custom_histories, time_name, t, id_name)

    covariate_fits, bounds, rmses, cov_model_coeffs, cov_model_stderrs, cov_model_vcovs, cov_model_fits_summary = \
        fit_covariate_model(covmodels=covmodels, covnames=covnames, covtypes=covtypes,
                            covfits_custom=covfits_custom, time_name=time_name, obs_data=resample_data,
                            return_fits=boot_diag, trunc_params=trunc_params, visit_names=visit_names,
                            max_visits=max_visits, ts_visit_names=ts_visit_names,
                            visit_covs=visit_covs, restrictions=restrictions)

    outcome_fit, outcome_model_coeffs, outcome_model_stderrs, outcome_model_vcovs, outcome_model_fits_summary = \
        fit_outcome_model(outcome_model=outcome_model, outcome_type=outcome_type,
                          outcome_name=outcome_name, time_name=time_name, obs_data=resample_data,
                          competing=competing, compevent_name=compevent_name, return_fits=boot_diag,
                          yrestrictions=yrestrictions)

    model_coeffs = {**cov_model_coeffs, **outcome_model_coeffs}
    model_stderrs = {**cov_model_stderrs, **outcome_model_stderrs}
    model_vcovs = {**cov_model_vcovs, **outcome_model_vcovs}
    model_fits_summary = {**cov_model_fits_summary, **outcome_model_fits_summary}

    if competing:
        compevent_fit, comp_model_coeffs, comp_model_stderrs, comp_model_vcovs, comp_model_fits_summary = \
            fit_compevent_model(compevent_model=compevent_model, compevent_name=compevent_name,
                                time_name=time_name, obs_data=resample_data, return_fits=boot_diag,
                                compevent_restrictions=compevent_restrictions)
        model_coeffs.update(comp_model_coeffs)
        model_stderrs.update(comp_model_stderrs)
        model_vcovs.update(comp_model_vcovs)
        model_fits_summary.update(comp_model_fits_summary)
    else:
        compevent_fit = None

    if n_simul != len(np.unique(resample_data[id_name])):
        data_list = dict(list(obs_data.groupby(id_name, group_keys=True)))
        ids = np.unique(obs_data[id_name])
        new_ids = np.random.choice(ids, n_simul, replace=True)

        new_df = []
        for index, new_id in enumerate(new_ids):
            new_id_df = data_list[new_id].copy()
            new_id_df[id_name] = index
            new_df.append(new_id_df)
        resample_data = pd.concat(new_df, ignore_index=True)

    boot_results = []
    boot_pools = []
    for i in range(len(int_descripts)):
        boot_result = simulate(seed=boot_seeds[boot_id], time_points=time_points, time_name=time_name,
                                   id_name=id_name, covnames=covnames, basecovs=basecovs,
                                   covmodels=covmodels,  covtypes=covtypes, cov_hist=cov_hist,
                                   covariate_fits=covariate_fits, rmses=rmses, bounds=bounds, outcome_type=outcome_type,
                                   obs_data=resample_data, intervention=interventions[i],
                                   intvar=intvars[i], int_time=int_times[i],
                                   custom_histvars = custom_histvars, custom_histories=custom_histories,
                                   covpredict_custom=covpredict_custom,
                                   outcome_fit=outcome_fit, outcome_name=outcome_name,
                                   competing=competing, compevent_name=compevent_name,
                                   compevent_fit=compevent_fit, compevent_model=compevent_model,
                                   compevent_cens=compevent_cens, trunc_params=trunc_params, visit_names=visit_names,
                                   visit_covs=visit_covs, ts_visit_names=ts_visit_names,
                                   max_visits=max_visits, time_thresholds=time_thresholds,
                                   baselags=baselags, below_zero_indicator=below_zero_indicator,
                                   restrictions=restrictions, yrestrictions=yrestrictions,
                                   compevent_restrictions=compevent_restrictions
                               )
        boot_results.append(boot_result['g_result'])
        boot_pools.append(boot_result['pool'])

    boot_results_dict = {'boot_results': boot_results, 'bootcoeffs': model_coeffs, 'bootstderrs': model_stderrs,
                         'bootvcovs': model_vcovs}

    if hazardratio:
        pool1 = boot_pools[intcomp[0]]
        pool2 = boot_pools[intcomp[1]]

        if competing and not compevent_cens:
            import cmprsk.cmprsk as cmprsk

            new_pool1 = pool1.groupby(id_name, group_keys=False).apply(hr_comp_data_helper,
                        outcome_name=outcome_name, compevent_name=compevent_name)
            new_pool2 = pool2.groupby(id_name, group_keys=False).apply(hr_comp_data_helper,
                        outcome_name=outcome_name, compevent_name=compevent_name)
            new_pool1['regime'] = 0
            new_pool2['regime'] = 1
            concat_data = pd.concat([new_pool1, new_pool2])
            concat_data = concat_data[[time_name, outcome_name, compevent_name, 'regime']]
            concat_data = concat_data.reset_index(drop=True)
            concat_data['event'] = np.where(concat_data[compevent_name] == 1, 2,
                                            concat_data[outcome_name]).tolist()
            ftime = concat_data[time_name]
            fstatus = concat_data['event']
            crr_res = cmprsk.crr(failure_time=ftime, failure_status=fstatus, static_covariates=concat_data[['regime']])
            hazard_ratio = crr_res.hazard_ratio()[0][0]
        else:
            new_pool1 = pool1.groupby(id_name, group_keys=False).apply(hr_data_helper, outcome_name=outcome_name)
            new_pool2 = pool2.groupby(id_name, group_keys=False).apply(hr_data_helper, outcome_name=outcome_name)
            new_pool1['regime'] = 0
            new_pool2['regime'] = 1
            concat_data = pd.concat([new_pool1, new_pool2])
            concat_data = concat_data[[time_name, outcome_name, 'regime']]
            cph = CoxPHFitter()
            cph.fit(concat_data, duration_col=time_name, event_col=outcome_name)
            hazard_ratio = cph.hazard_ratios_.values[0]

        boot_results_dict['boot_hr'] = hazard_ratio

    return boot_results_dict


