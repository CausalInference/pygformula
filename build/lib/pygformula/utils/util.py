import numpy as np
import json
import os
import re
import prettytable as pt
import pandas as pd
from scipy import stats


def read_intervention_input(interventions, int_descript):

    intervention_dicts = {}
    for intervention_key, intervention in interventions.items():
        prefix = intervention_key.split('_', 1)[0]  # Get the prefix (e.g., 'Intervention1', 'Intervention2')
        intervention_id = int(prefix[12:])  # Get id
        intervention_name = int_descript[intervention_id - 1]
        index = intervention_key.find('_') # Get the treatment name
        treatment_name = intervention_key[index+1:]

        intervention.insert(0, treatment_name)
        if intervention_name not in intervention_dicts:
            intervention_dicts[intervention_name] = []
        intervention_dicts[intervention_name].append(intervention)

    return intervention_dicts


def error_catch(obs_data, id, time_points, interventions, intervention_dicts, int_descript, custom_histvars,
                custom_histories, covfits_custom, covpredict_custom, time_name, outcome_name, censor_name, censor_model,
                ipw_cutoff_quantile, ipw_cutoff_value, outcome_type, ref_int, covnames = None, covtypes = None,
                covmodels=None, ymodel=None, compevent_name=None, compevent_model=None, intcomp=None,
                trunc_params=None, basecovs=None, time_thresholds=None):

    """
    This is an internal function to catch various potential errors in the user input.

    Parameters
    ----------
    obs_data: DataFrame
        A data frame containing the observed data.

    id: Str
        A string specifying the name of the id variable in obs_data.

    time_points: Int
        An integer indicating the number of time points to simulate. It is set equal to the maximum number of records
        that obs_data contains for any individual plus 1, if not specified by users.

    interventions: Dict
        A dictionary whose key is the treatment name in the intervention with the format Intervention{id}_{treatment_name},
        value is a list that contains the intervention function, values required by the function, and a list of time
        points in which the intervention is applied.

    intervention_dicts: Dict
        A dictionary whose key is the intervention decription and the value is the intervention list for all treatment
        variables in this intervention.

    int_descript: List
        A list of strings, each of which describes a user-specified intervention.

    custom_histvars: List
        A list of strings, each of which specifies the names of the time-varying covariates with user-specified custom histories.

    custom_histories: List
        A list of functions, each function is the user-specified custom history functions for covariates. The list
        should be the same length as custom_histvars and in the same order.

    covfits_custom: List
        A list, each element could be 'NA' or a user-specified fit function. The non-NA value is set
        for the covariates with custom type. The 'NA' value is set for other covariates. The list must be the
        same length as covnames and in the same order.

    covpredict_custom: List
        A list, each element could be 'NA' or a user-specified predict function. The non-NA value is set
        for the covariates with custom type. The 'NA' value is set for other covariates. The list must be the
        same length as covnames and in the same order.

    time_name: Str
        A string specifying the name of the time variable in obs_data.

    outcome_name: Str
        A string specifying the name of the outcome variable in obs_data.

    covnames: List
        A list of strings specifying the names of the time-varying covariates in obs_data.

    covtypes: List
        A list of strings specifying the “type” of each time-varying covariate included in covnames. The supported types:
        "binary", "normal", "categorical", "bounded normal", "zero-inflated normal", "truncated normal", "absorbing",
        "categorical time", "square time" and "custom". The list must be the same length as covnames and in the same order.

    censor_name: Str
        A string specifying the name of the censoring variable in obs_data. Only applicable when using inverse probability
        weights to estimate the natural course means / risk from the observed data.

    censor_model: Str
        A string specifying the model statement for the censoring variable. Only applicable when using inverse probability
         weights to estimate the natural course means / risk from the observed data.

    ipw_cutoff_quantile: Float
        Percentile value for truncation of the inverse probability weights.

    ipw_cutoff_value: Float
        Absolute value for truncation of the inverse probability weights.

    outcome_type: Str
        A string specifying the "type" of outcome. The possible "types" are: "survival", "continuous_eof", and "binary_eof".

    ref_int: Int
        An integer indicating the intervention to be used as the reference for calculating the end-of-follow-up mean ratio
        and mean difference. 0 denotes the natural course, while subsequent integers denote user-specified interventions
        in the order that they are named in interventions. It is set to 0 if not specified by users.

    covmodels: List
        A list of strings, where each string is the model statement of the time-varying covariate. The list must be the
        same length as covnames and in the same order. If a model is not required for a certain covariate, it should be
        set to 'NA' at that index.

    ymodel: Str
        A string specifying the model statement for the outcome variable.

    compevent_name: Str
        A string specifying the name of the competing event variable in obs_data. Only applicable for survival outcomes.

    compevent_model: Str
        A string specifying the model statement for the competing event variable. Only applicable for survival outcomes.

    intcomp: List
        List of two numbers indicating a pair of interventions to be compared by a hazard ratio.

    trunc_params: List
        A list, each element could be 'NA' or a two-element list. If not 'NA', the first element specifies the truncated
        value and the second element specifies the truncated direction (‘left’ or ‘right’). The non-NA value is set
        for the truncated normal covariates. The 'NA' value is set for other covariates. The list should be the same
        length as covnames and in the same order.

    basecovs: List
        A list of strings specifying the names of baseline covariates in obs_data. These covariates should not be
        included in covnames.

    time_thresholds: List
        A list of integers that splits the time points into different intervals. It is used to create the variable
        "categorical time".

    Returns
    -------
    None

    """

    if obs_data is None:
        raise ValueError('Missing input observational data.')
    if not isinstance(obs_data, pd.DataFrame):
        raise TypeError('obs_data should be DataFrame type.')
    if id not in obs_data.columns:
        raise ValueError('Missing id name in the observational data.')
    if time_name not in obs_data.columns:
        raise ValueError('Missing time name in the observational data.')
    if outcome_name not in obs_data.columns:
        raise ValueError('Missing outcome name in the observational data.')
    if ymodel is None:
        raise ValueError('Missing outcome model.')
    if outcome_type not in ['survival', 'continuous_eof', 'binary_eof']:
        raise ValueError('Please specify the outcome type as survival, continuous_eof, or binary_eof.')

    if outcome_type == 'survival':
        if time_points > np.max(np.unique(obs_data[time_name])) + 1:
            raise ValueError('Number of simulated time points should be set to a value within the observed follow-up the data.')

    if outcome_type == 'continuous_eof' or outcome_type == 'binary_eof':
        if time_points != np.max(np.unique(obs_data[time_name])) + 1:
            raise ValueError('For end of follow up outcomes, the mean is calculated at the last time point in obs_data.'
                             'The value of argument time_points should be the same length of the data record for each individual plus 1.')

    if censor_name is None and censor_model is not None:
        raise ValueError('The censor_name should be specified when there is a censor model.')
    if censor_model is None and censor_name is not None:
        raise ValueError('The censor_model should be specified when there is a censor name.')

    if ipw_cutoff_quantile:
        if ipw_cutoff_quantile < 0 or ipw_cutoff_quantile > 1:
            raise ValueError('The ipw_cutoff_quantile should be a value between 0 and 1.')
    if ipw_cutoff_value:
        if ipw_cutoff_value < 0 or ipw_cutoff_value > 1:
            raise ValueError('The ipw_cutoff_value should be a value between 0 and 1.')

    if compevent_name is None and compevent_model is not None:
        raise ValueError('The compevent_name should be specified when there is a competing model.')
    if compevent_model is None and compevent_name is not None:
        raise ValueError('The compevent_model should be specified when there is a competing name.')

    if basecovs is not None:
        for basecov in basecovs:
            unique_num = obs_data.groupby(id)[basecov].nunique().tolist()
            if len(set(unique_num)) != 1:
                raise ValueError('The baseline covariate for each individual should be the same value at all time steps.')

    if covnames is not None:
        for k, covname in enumerate(covnames):
            if covtypes[k] != 'categorical time' and covtypes[k] != 'square time' and covname not in obs_data.columns:
                raise ValueError('Missing covariate name in the observational data.')
        if not len(covnames) == len(covtypes) == len(covmodels):
            raise ValueError('covnames, covtypes and covmodels are unequal lengths.')

    if custom_histvars is not None:
        if len(custom_histvars) != len(custom_histories):
            raise ValueError('custom_histvars and custom_histories should have the same length.')
        for histvar in custom_histvars:
            if histvar not in covnames:
                raise ValueError('The variable in custom_histvars is not found in covnames.')

    if covfits_custom is not None:
        if len(covfits_custom) != len(covpredict_custom):
            raise ValueError('covfits_custom and covpredict_custom should have the same length.')

    all_covtypes = ['binary', 'normal', 'categorical', 'bounded normal', 'zero-inflated normal', 'truncated normal',
                    'absorbing', 'categorical time', 'square time', 'custom']

    if covtypes is not None:
        for k, covtype in enumerate(covtypes):
            if covtype not in all_covtypes:
                raise ValueError('The input covariate type is not supported.')
            if covtype == 'truncated normal':
                if trunc_params is None:
                    raise ValueError('The truncated variable should contain truncated direction and value.')
                if not isinstance(trunc_params[k][0], (int, float)):
                    raise ValueError('The truncated value should be numeric.')
                if trunc_params[k][1] not in ['left', 'right']:
                    raise ValueError('The truncated direction should be left or right.')
            if covtype == 'binary':
                if not all([v == 1 or v == 0 for v in set(obs_data[covnames[k]])]):
                    raise ValueError('Binary data contains value other than 0 and 1.')
            if covtype == 'categorical time':
                if time_thresholds is None:
                    raise ValueError('The time_thresholds should be specified when there is categorical time variable.')

    if compevent_name is not None:
        if compevent_name not in obs_data.columns:
            raise ValueError('Missing competing name in the observational data.')

    for intervention_id, intervention in interventions.items():
        if not intervention_id.startswith('Intervention'):
            raise ValueError('The name {0} is not recognized, please specify it as the correct format'.format(intervention_id))

    if int_descript is not None:
        if not len(int_descript) == len(intervention_dicts.items()):
            raise ValueError('The number of specified interventions in argument int_descript and interventions is different.')

    if intcomp is not None:
        if not isinstance(intcomp[0], int):
            raise ValueError('The value in intcomp for comparison should be an integer that indicates the intervention.')
        if not isinstance(intcomp[1], int):
            raise ValueError('The value in intcomp for comparison should be an integer that indicates the intervention.')

    if not isinstance(ref_int, int):
        raise ValueError('Parameter ref_int should be an integer that indicates the desired reference intervention.')


def keywords_check(interventions):
    for key in interventions.keys():
        if 'Intervention' in key:
            pattern = r'^Intervention\d+_.+$'
            if not bool(re.match(pattern, key)):
                raise ValueError('The key {0} for specifying intervention is not found, please specify the name'
                                 ' following the correct format.'.format(key))
        else:
            raise ValueError('The key {0} for specifying intervention is invalid, please specify the intervention'
                             ' following the correct format'.format(key))

def get_output(ref_int, int_descript, censor, obs_res, g_results, time_points, ci_method, time_name, obs_means,
               outcome_type, nsamples, boot_results=None):
    """
    An internal function to get and print the output of the g-formula.

    Parameters
    ----------
    ref_int: Int
        An integer indicating the intervention to be used as the reference for calculating the end-of-follow-up mean
        ratio and mean difference. 0 denotes the natural course, while subsequent integers denote user-specified
        interventions in the order that they are named in interventions. It is set to 0 if not specified by users.

    int_descript: List
        A list of strings, describing the natural course intervention (in the first) and user-specified intervention(s).

    censor: Bool
        A boolean value indicating if there is a censoring event.

    obs_res: Float
        The nonparametric risk value at the final time point.

    g_results: List
        List of lists. The kth inner list contains the parametric estimate of risk at all the time points for the kth intervention.

    time_points: Int
        An integer indicating the number of time points to simulate. It is set equal to the maximum number of records
        that obs_data contains for any individual plus 1, if not specified by users.

    ci_method: Str
        A string specifying the method for calculating the bootstrap 95% confidence intervals, if applicable. The options
        are "percentile" and "normal". It is set to "percentile" if not specified by users.

    time_name: Str
        A string specifying the name of the time variable in obs_data.

    obs_means: Dict
        A dictionary, where the key is the covariate / risk name and the value is its observational mean at all the time points.

    outcome_type: Str
        A string specifying the "type" of outcome. The possible "types" are: "survival", "continuous_eof", and "binary_eof".

    nsamples: Int
        An integer specifying the number of bootstrap samples to generate.

    boot_results: List, default is None
        Three-dimensional list that stores the parametric risk estimates (survial outcome) or parametric outcome mean
        (for end-of-follow-up outcomes), the first dimension is the number of bootstrap samples, the second dimension is
        the number of interventions (natural course included), the third dimension is the time points.

    Returns
    -------
    res_table: DataFrame
        A dataframe contains the nonparametric and parametric estimates for natural course and all the user-specified interventions.

    """

    if outcome_type == 'survival':
        all_simulate_results = np.array(g_results)
        all_ref_result = all_simulate_results[ref_int]
        all_result_diff = all_simulate_results - all_ref_result
        all_result_ratio = all_simulate_results / all_ref_result

        simulate_results = all_simulate_results[:, time_points - 1]
        result_diff = all_result_diff[:, time_points - 1]
        result_ratio = all_result_ratio[:, time_points - 1]

        if nsamples == 0:
            # save nonparametric and parametric risks at all the time points to a data table
            table = pd.DataFrame(
                {'g-form risk (NICE-parametric)': all_simulate_results.transpose((1, 0)).flatten(),
                 'Risk Ratio': all_result_ratio.transpose((1, 0)).flatten(),
                 'Risk Difference': all_result_diff.transpose((1, 0)).flatten()})

            table.insert(0, time_name, value=np.array([[int(t) for k in range(len(int_descript))]
                                                       for t in range(time_points)]).flatten())
            table.insert(1, 'Intervention',
                         value=np.array([[name for name in int_descript] for t in range(time_points)]).flatten())
            table.insert(2, 'IP weighted risk' if censor else 'NP risk',
                         value=np.array(
                             [[obs_means['risk'][t] if i == 0 else 'NA' for i in range(len(int_descript))]
                              for t in range(time_points)]).flatten())
            res_table = table

            # output nonparametric and parametric risks at the specified final time point
            output = pt.PrettyTable()
            output.add_column('Intervention', [int_name if i != ref_int else ''.join([int_name, '(ref)'])
                                               for i, int_name in enumerate(int_descript)])
            output.add_column('IP weighted risk' if censor else 'NP risk',
                              ['{:.5f}'.format(obs_res) if int_name == 'Natural course' else 'NA'
                               for i, int_name in enumerate(int_descript)])
            output.add_column('g-formula risk (NICE-parametric)',
                              ['{:.5f}'.format(simulate_results[i]) for i in range(len(int_descript))])
            output.add_column('Risk Ratio(RR)',
                              ['{:.5f}'.format(result_ratio[i]) for i in range(len(int_descript))])
            output.add_column('Risk Difference(RD)',
                              ['{:.5f}'.format(result_diff[i]) for i in range(len(int_descript))])
            print(output)

        else:
            all_boot_risk_std = np.std(np.array(boot_results), ddof=1, axis=0)
            all_boot_risk_difference = \
                np.array([np.array(boot_results)[:, i, :] - np.array(boot_results)[:, ref_int, :]
                          for i in range(len(int_descript))])
            all_boot_rd_std = np.std(all_boot_risk_difference, ddof=1, axis=1)

            all_boot_risk_ratio = \
                np.array([np.array(boot_results)[:, i, :] / np.array(boot_results)[:, ref_int, :]
                          for i in range(len(int_descript))])
            all_boot_rr_std = np.std(all_boot_risk_ratio, ddof=1, axis=1)

            boot_risk_std = all_boot_risk_std[:, time_points - 1]
            boot_rd_std = all_boot_rd_std[:, time_points - 1]
            boot_rr_std = all_boot_rr_std[:, time_points - 1]

            if ci_method == 'percentile':
                all_risk_lb = np.percentile(np.array(boot_results), 2.5, axis=0)
                all_risk_ub = np.percentile(np.array(boot_results), 97.5, axis=0)
                risk_lb = all_risk_lb[:, time_points - 1]
                risk_ub = all_risk_ub[:, time_points - 1]

                all_boot_rd_lb = np.percentile(all_boot_risk_difference, 2.5, axis=1)
                all_boot_rd_ub = np.percentile(all_boot_risk_difference, 97.5, axis=1)
                boot_rd_lb = all_boot_rd_lb[:, time_points - 1]
                boot_rd_ub = all_boot_rd_ub[:, time_points - 1]

                all_boot_rr_lb = np.percentile(all_boot_risk_ratio, 2.5, axis=1)
                all_boot_rr_ub = np.percentile(all_boot_risk_ratio, 97.5, axis=1)
                boot_rr_lb = all_boot_rr_lb[:, time_points - 1]
                boot_rr_ub = all_boot_rr_ub[:, time_points - 1]

            elif ci_method == 'normal':
                all_boot_risk_ci = \
                    np.array([stats.norm.interval(0.95, loc=all_simulate_results[i, :], scale=all_boot_risk_std[i, :])
                              for i in range(len(int_descript))])
                all_risk_lb = all_boot_risk_ci[:, 0, :]
                all_risk_ub = all_boot_risk_ci[:, 1, :]
                risk_lb = all_risk_lb[:, time_points - 1]
                risk_ub = all_risk_ub[:, time_points - 1]

                all_boot_rd_std = np.std(all_boot_risk_difference, ddof=1, axis=1)
                all_boot_rd_ci = \
                    np.array([stats.norm.interval(0.95, loc=all_result_diff[i, :], scale=all_boot_rd_std[i, :])
                              if i != ref_int else [np.zeros(time_points), np.zeros(time_points)]
                              for i in range(len(int_descript))])

                all_boot_rd_lb = all_boot_rd_ci[:, 0, :]
                all_boot_rd_ub = all_boot_rd_ci[:, 1, :]
                boot_rd_lb = all_boot_rd_lb[:, time_points - 1]
                boot_rd_ub = all_boot_rd_ub[:, time_points - 1]

                all_boot_rr_std = np.std(all_boot_risk_ratio, ddof=1, axis=1)
                all_boot_rr_ci = \
                    np.array([stats.norm.interval(0.95, loc=all_result_ratio[i, :], scale=all_boot_rr_std[i, :])
                              if i != ref_int else [np.ones(time_points), np.ones(time_points)]
                              for i in range(len(int_descript))])

                all_boot_rr_lb = all_boot_rr_ci[:, 0, :]
                all_boot_rr_ub = all_boot_rr_ci[:, 1, :]
                boot_rr_lb = all_boot_rr_lb[:, time_points - 1]
                boot_rr_ub = all_boot_rr_ub[:, time_points - 1]

            else:
                raise ValueError("The input method is not found, the options for calculating the bootstrap confidence "
                                 "intervals are 'percentile' and 'normal'.")

            # save nonparametric and parametric risks with bootstrap confidence intervals at all time points to a data table
            boot_table = pd.DataFrame(
                {'g-form risk (NICE-parametric)': all_simulate_results.transpose((1, 0)).flatten(),
                 'Risk Standard Error': all_boot_risk_std.transpose((1, 0)).flatten(),
                 'Risk 95% lower bound': all_risk_lb.transpose((1, 0)).flatten(),
                 'Risk 95% upper bound': all_risk_ub.transpose((1, 0)).flatten(),
                 'Risk Ratio': all_result_ratio.transpose((1, 0)).flatten(),
                 'RR SE': all_boot_rr_std.transpose((1, 0)).flatten(),
                 'RR 95% lower bound': all_boot_rr_lb.transpose((1, 0)).flatten(),
                 'RR 95% upper bound': all_boot_rr_ub.transpose((1, 0)).flatten(),
                 'Risk Difference': all_result_diff.transpose((1, 0)).flatten(),
                 'RD SE': all_boot_rd_std.transpose((1, 0)).flatten(),
                 'RD 95% lower bound': all_boot_rd_lb.transpose((1, 0)).flatten(),
                 'RD 95% upper bound': all_boot_rd_ub.transpose((1, 0)).flatten()})
            boot_table.insert(0, time_name, value=np.array(
                [[int(t) for k in range(len(int_descript))] for t in
                 range(time_points)]).flatten())
            boot_table.insert(1, 'Intervention', value=np.array(
                [[name for name in int_descript] for t in range(time_points)]).flatten())
            boot_table.insert(2, 'IP weighted mean' if censor else 'NP mean', value=np.array(
                [[obs_means['risk'][t] if i == 0 else 'NA' for i in range(len(int_descript))]
                 for t in range(time_points)]).flatten())
            res_table = boot_table

            # output nonparametric and parametric risks with bootstrap confidence intervals at the specified final time point
            output = pt.PrettyTable()
            output.add_column('Intervention',
                              [intervention_name if i != ref_int else ''.join([intervention_name, '(ref)'])
                               for i, intervention_name in enumerate(int_descript)])

            output.add_column('IP weighted risk' if censor else 'NP risk',
                              ['{:.5f}'.format(obs_res) if int_name == 'Natural course' else 'NA'
                               for i, int_name in enumerate(int_descript)])
            output.add_column('g-formula risk (NICE-parametric)',
                              ['{:.5f}'.format(simulate_results[i]) for i in range(len(int_descript))])
            output.add_column('Risk Standard Error',
                              ['{:.5f}'.format(boot_risk_std[i]) for i in range(len(int_descript))])
            output.add_column('Risk 95% lower bound',
                              ['{:.5f}'.format(risk_lb[i]) for i in range(len(int_descript))])
            output.add_column('Risk 95% upper bound',
                              ['{:.5f}'.format(risk_ub[i]) for i in range(len(int_descript))])
            output.add_column('Risk Ratio(RR)',
                              ['{:.5f}'.format(result_ratio[i]) for i in range(len(int_descript))])
            output.add_column('RR SE',
                              ['{:.5f}'.format(boot_rr_std[i]) for i in range(len(int_descript))])
            output.add_column('RR 95% lower bound',
                              ['{:.5f}'.format(boot_rr_lb[i]) for i in range(len(int_descript))])
            output.add_column('RR 95% upper bound',
                              ['{:.5f}'.format(boot_rr_ub[i]) for i in range(len(int_descript))])
            output.add_column('Risk Difference(RD)',
                              ['{:.5f}'.format(result_diff[i]) for i in range(len(int_descript))])
            output.add_column('RD SE',
                              ['{:.5f}'.format(boot_rd_std[i]) for i in range(len(int_descript))])
            output.add_column('RD 95% lower bound',
                              ['{:.5f}'.format(boot_rd_lb[i]) for i in range(len(int_descript))])
            output.add_column('RD 95% upper bound',
                              ['{:.5f}'.format(boot_rd_ub[i]) for i in range(len(int_descript))])

            output_res = output.get_string(
                fields=['Intervention', 'IP weighted risk' if censor else 'NP risk', 'g-formula risk (NICE-parametric)',
                        'Risk Standard Error', 'Risk 95% lower bound', 'Risk 95% upper bound'])
            output_rr_rd = output.get_string(
                fields=['Risk Ratio(RR)', 'RR SE', 'RR 95% lower bound', 'RR 95% upper bound',
                        'Risk Difference(RD)', 'RD SE', 'RD 95% lower bound', 'RD 95% upper bound'])
            print(output_res)
            print(output_rr_rd)

    elif outcome_type == 'binary_eof' or outcome_type == 'continuous_eof':
        simulate_results = np.array(g_results)
        ref_result = simulate_results[ref_int]
        result_diff = simulate_results - ref_result
        result_ratio = simulate_results / ref_result

        if nsamples == 0:
            # save nonparametric and parametric estimate at final time point to a data table
            table = pd.DataFrame(
                {'g-form mean (NICE)': simulate_results, 'Mean ratio': result_ratio, 'Mean difference': result_diff})
            table.insert(0, time_name, value=time_points - 1)
            table.insert(1, 'Intervention', value=np.array(int_descript))
            table.insert(2, 'IP weighted mean' if censor else 'NP mean',
                         value=np.array([obs_res if i == 0 else 'NA' for i in range(len(int_descript))]))
            res_table = table

            # output nonparametric and parametric estimate at final time point
            output = pt.PrettyTable()
            output.add_column('Intervention', [int_name if i != ref_int else ''.join([int_name, '(ref)'])
                                               for i, int_name in enumerate(int_descript)])
            output.add_column('IP weighted mean' if censor else 'NP mean',
                              ['{:.5f}'.format(obs_res) if int_name == 'Natural course' else 'NA'
                               for i, int_name in enumerate(int_descript)])
            output.add_column('g-formula mean (NICE-parametric)',
                              ['{:.5f}'.format(simulate_results[i]) for i in range(len(int_descript))])
            output.add_column('Mean Ratio(MR)',
                              ['{:.5f}'.format(result_ratio[i]) for i in range(len(int_descript))])
            output.add_column('Mean Difference(MD)',
                              ['{:.5f}'.format(result_diff[i]) for i in range(len(int_descript))])
            print(output)

        else:
            boot_mean_results = np.array(boot_results)
            boot_mean_std = np.std(boot_mean_results, ddof=1, axis=0)

            boot_mean_difference = np.array([boot_mean_results[:, i] - boot_mean_results[:, ref_int]
                                             for i in range(len(int_descript))])
            boot_md_std = np.std(boot_mean_difference, ddof=1, axis=1)

            boot_mean_ratio = np.array([boot_mean_results[:, i] / boot_mean_results[:, ref_int]
                                        for i in range(len(int_descript))])
            boot_mr_std = np.std(boot_mean_ratio, ddof=1, axis=1)

            if ci_method == 'percentile':
                mean_lb = np.percentile(boot_mean_results, 2.5, axis=0)
                mean_ub = np.percentile(boot_mean_results, 97.5, axis=0)

                boot_md_lb = np.percentile(boot_mean_difference, 2.5, axis=1)
                boot_md_ub = np.percentile(boot_mean_difference, 97.5, axis=1)

                boot_mr_lb = np.percentile(boot_mean_ratio, 2.5, axis=1)
                boot_mr_ub = np.percentile(boot_mean_ratio, 97.5, axis=1)

            elif ci_method == 'normal':
                boot_mean_ci = \
                    np.array([stats.norm.interval(0.95, loc=simulate_results[i], scale=boot_mean_std[i])
                              for i in range(len(int_descript))])
                mean_lb, mean_ub = boot_mean_ci[:, 0], boot_mean_ci[:, 1]

                boot_md_std = np.std(boot_mean_difference, ddof=1, axis=1)
                boot_md_ci = \
                    np.array([stats.norm.interval(0.95, loc=result_diff[i], scale=boot_md_std[i])
                              if i != ref_int else (0.0, 0.0) for i in range(len(int_descript))])
                boot_md_lb, boot_md_ub = boot_md_ci[:, 0], boot_md_ci[:, 1]

                boot_mr_std = np.std(boot_mean_ratio, ddof=1, axis=1)
                boot_mr_ci = \
                    np.array([stats.norm.interval(0.95, loc=result_ratio[i], scale=boot_mr_std[i])
                              if i != ref_int else (1.0, 1.0) for i in range(len(int_descript))])
                boot_mr_lb, boot_mr_ub = boot_mr_ci[:, 0], boot_mr_ci[:, 1]

            else:
                raise ValueError("The input method is not found, the options for calculating the bootstrap confidence "
                                 "intervals are 'percentile' and 'normal'.")

            # save nonparametric and parametric estimate with bootstrap confidence intervals at final time point to a data table
            boot_table = pd.DataFrame(
                {'g-form mean (NICE-parametric)': simulate_results, 'Mean Standard Error': boot_mean_std,
                 'Mean 95% lower bound': mean_lb,
                 'Mean 95% upper bound': mean_ub,
                 'Mean Ratio(MR)': result_ratio,
                 'MR SE': boot_mr_std,
                 'MR 95% lower bound': boot_mr_lb,
                 'MR 95% upper bound': boot_mr_ub,
                 'Mean Difference(MD)': result_diff,
                 'MD SE': boot_md_std,
                 'MD 95% lower bound': boot_md_lb,
                 'MD 95% upper bound': boot_md_ub
                 })
            boot_table.insert(0, time_name, value=time_points - 1)
            boot_table.insert(1, 'Intervention', value=int_descript)
            boot_table.insert(2, 'IP weighted mean' if censor else 'NP mean', value=np.array(
                [obs_res if i == 0 else 'NA' for i in range(len(int_descript))]))
            res_table = boot_table

            # output nonparametric and parametric estimate with bootstrap confidence intervals at final time point
            output = pt.PrettyTable()
            output.add_column('Intervention',
                              [intervention_name if i != ref_int else ''.join([intervention_name, '(ref)'])
                               for i, intervention_name in enumerate(int_descript)])

            output.add_column('IP weighted mean' if censor else 'NP mean',
                              ['{:.5f}'.format(obs_res) if int_name == 'Natural course' else 'NA'
                               for i, int_name in enumerate(int_descript)])
            output.add_column('g-formula mean (NICE-parametric)',
                              ['{:.5f}'.format(simulate_results[i]) for i in range(len(int_descript))])
            output.add_column('Mean Standard Error',
                              ['{:.5f}'.format(boot_mean_std[i]) for i in range(len(int_descript))])
            output.add_column('Mean 95% lower bound',
                              ['{:.5f}'.format(mean_lb[i]) for i in range(len(int_descript))])
            output.add_column('Mean 95% upper bound',
                              ['{:.5f}'.format(mean_ub[i]) for i in range(len(int_descript))])
            output.add_column('Mean Ratio(MR)',
                              ['{:.5f}'.format(result_ratio[i]) for i in range(len(int_descript))])
            output.add_column('MR SE',
                              ['{:.5f}'.format(boot_mr_std[i]) for i in range(len(int_descript))])
            output.add_column('MR 95% lower bound',
                              ['{:.5f}'.format(boot_mr_lb[i]) for i in range(len(int_descript))])
            output.add_column('MR 95% upper bound',
                              ['{:.5f}'.format(boot_mr_ub[i]) for i in range(len(int_descript))])
            output.add_column('Mean Difference(MD)',
                              ['{:.5f}'.format(result_diff[i]) for i in range(len(int_descript))])
            output.add_column('MD SE',
                              ['{:.5f}'.format(boot_md_std[i]) for i in range(len(int_descript))])
            output.add_column('MD 95% lower bound',
                              ['{:.5f}'.format(boot_md_lb[i]) for i in range(len(int_descript))])
            output.add_column('MD 95% upper bound',
                              ['{:.5f}'.format(boot_md_ub[i]) for i in range(len(int_descript))])

            output_res = output.get_string(
                fields=['Intervention', 'IP weighted mean' if censor else 'NP mean',
                        'g-formula mean (NICE-parametric)', 'Mean Standard Error', 'Mean 95% lower bound',
                        'Mean 95% upper bound'])
            output_mr_md = output.get_string(
                fields=['Mean Ratio(MR)', 'MR SE', 'MR 95% lower bound', 'MR 95% upper bound',
                        'Mean Difference(MD)', 'MD SE', 'MD 95% lower bound', 'MD 95% upper bound'])
            print(output_res)
            print(output_mr_md)

    return res_table


def get_hr_output(boot_results_dicts, nsamples, ci_method, hazard_ratio):
    """
    This is an inernal function to print the output of hazard ratio when the bootstrap samples is not 0.

    Parameters
    ----------
    boot_results_dicts: List
        A list of dictionaries, each dictionary contains the 'boot_results', 'bootcoeffs', 'bootstderrs', 'bootvcovs',
        'boot_hr' for each bootstrap sample.

    nsamples: Int
        An integer specifying the number of bootstrap samples to generate.

    ci_method: Str
        A string specifying the method for calculating the bootstrap 95% confidence intervals, if applicable. The options
        are "percentile" and "normal". It is set to "percentile" if not specified by users.

    hazard_ratio: Float
        The hazard ratio of the two compared interventions on the observational data.

    Returns
    -------
    None

    """
    boot_hr_results = [boot_results_dicts[i]['boot_hr'] for i in range(nsamples) if boot_results_dicts[i]['boot_hr'] is not None]
    hr_se = np.std(boot_hr_results, ddof=1)
    if ci_method == 'percentile':
        lb_hr = np.percentile(boot_hr_results, 2.5)
        ub_hr = np.percentile(boot_hr_results, 97.5)
    elif ci_method == 'normal':
        boot_hr_ci = stats.norm.interval(0.95, loc=hazard_ratio, scale=hr_se)
        lb_hr = boot_hr_ci[0]
        ub_hr = boot_hr_ci[1]
    boot_hr = pt.PrettyTable()
    boot_hr.add_column('Hazardratio (HR)', ['{:.5f}'.format(hazard_ratio)])
    boot_hr.add_column('HR SE', ['{:.5f}'.format(hr_se)])
    boot_hr.add_column('HR 95% lower bound', ['{:.5f}'.format(lb_hr)])
    boot_hr.add_column('HR 95% upper bound', ['{:.5f}'.format(ub_hr)])
    boot_hr = boot_hr
    print(boot_hr)
    

def save_config(save_path, **config_parameters):
    config_dict = {}
    for key, value in config_parameters.items():
        if isinstance(value, np.int64):
            config_dict[key] = int(value)
        else:
            config_dict[key] = value
    config_dict = json.dumps(config_dict, indent=1)
    config_path = os.path.join(save_path, 'config_dict.json')
    with open(config_path, 'w') as f:
        f.write(config_dict)
    f.close()


def save_results(summary_dict, save_path):
    summary_dict['gformula_results'].to_csv(os.path.join(save_path, 'gformula_results.csv'))
    sim_data_path = os.path.join(save_path, 'sim_data')
    if not os.path.exists(sim_data_path):
        os.makedirs(sim_data_path)
    for name, sim_data in summary_dict['sim_data'].items():
        sim_data.to_csv(os.path.join(sim_data_path, 'sim_data_{0}.csv'.format(name)))
    f = open(os.path.join(save_path, 'results.txt'), 'w')
    for k, v in summary_dict.items():
        if k != 'gformula_results' and k != 'sim_data':
            f.write(k + ':' + str(v))
            f.write('\n')
    f.close()


