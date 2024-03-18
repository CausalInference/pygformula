import numpy as np
import pandas as pd
import random
import time
import os
from joblib import Parallel, delayed
from tqdm import tqdm
from lifelines import CoxPHFitter
from .fit import fit_covariate_model, fit_outcome_model, fit_compevent_model, fit_censor_model
from .histories import update_precoded_history, update_custom_history
from .simulate import simulate
from .bootstrap import Bootstrap
from .interventions import natural
from ..utils.helper import get_cov_hist_info, visit_func, hr_data_helper, hr_comp_data_helper, categorical_func
from ..utils.util import read_intervention_input, error_catch, save_config, save_results, get_output, get_hr_output
from ..comparisons import comparison_calculate
from ..plot import plot_natural_course, plot_interventions


class ParametricGformula:
    """
    G-formula estimation under parametric models.

    Parameters
    ----------
    obs_data: DataFrame
        A data frame containing the observed data.

    id_name: Str
        A string specifying the name of the id variable in obs_data.

    time_name: Str
        A string specifying the name of the time variable in obs_data.

    outcome_name: Str
         A string specifying the name of the outcome variable in obs_data.

    outcome_model: Str
        A string specifying the model statement for the outcome variable.

    covnames: List, default is None
         A list of strings specifying the names of the time-varying covariates in obs_data.

    covtypes: List, default is None
        A list of strings specifying the “type” of each time-varying covariate included in covnames.
        The supported types: "binary", "normal", "categorical", "bounded normal", "zero-inflated normal",
        "truncated normal", "absorbing", "categorical time", "square time" and "custom". The list must be the same
        length as covnames and in the same order.

    covmodels: List, default is None
        A list of strings, where each string is the model statement of the time-varying covariate. The list
        must be the same length as covnames and in the same order. If a model is not required for a certain covariate,
        it should be set to 'NA' at that index.

    int_descript: List, default is None
        A list of strings, each of which describes a user-specified intervention.

    custom_histvars: List, default is None
        A list of strings, each of which specifies the names of the time-varying covariates with user-specified custom histories.

    custom_histories: List, default is None
        A list of function, each function is the user-specified custom history functions for covariates. The list
        must be the same length as custom_histvars and in the same order.

    covfits_custom: List, default is None
        A list, at the index where the covtype is set to "custom", the element is a user-specified fit function,
        otherwise it should be set to 'NA'. The list must be the same length as covnames and in the same order.

    covpredict_custom: List, default is None
        A list, at the index where the covtype is set to "custom", the element is a user-specified predict function,
        otherwise it should be set to 'NA'.

    nsamples: Int, default is 0.
        An integer specifying the number of bootstrap samples to generate.

    compevent_name: Str, default is None
        A string specifying the name of the competing event variable in obs_data. Only applicable for survival outcomes.

    compevent_model: Str, default is None
        A string specifying the model statement for the competing event variable. Only applicable for survival outcomes.

    compevent_cens: Bool, default is False
        A boolean value indicating whether to treat competing events as censoring events.

    intcomp: List, default is None
        List of two numbers indicating a pair of interventions to be compared by a hazard ratio.

    censor_name: Str, default is None
        A string specifying the name of the censoring variable in obs_data. Only applicable when using inverse
        probability weights to estimate the natural course means / risk from the observed data.

    censor_model: Str, default is None
        A string specifying the model statement for the censoring variable. Only applicable when using inverse
        probability weights to estimate the natural course means / risk from the observed data.

    model_fits: Bool, default is False
        A boolean value indicating whether to return the fitted models.

    boot_diag: Bool, default is False
        A boolean value indicating whether to return the parametric g-formula estimates as well as the coefficients,
        standard errors, and variance-covariance matrices of the parameters of the fitted models in the bootstrap samples.

    ipw_cutoff_quantile: Float, default is None
        A percentile by which to truncate inverse probability weights.

    ipw_cutoff_value: Float, default is None
        A cutoff value by which to truncate inverse probability weights.

    outcome_type: Str, default is None
        A string specifying the "type" of outcome. The possible "types" are: "survival", "continuous_eof", and "binary_eof".

    trunc_params: List, default is None
        A list, at the index where the covtype is set to "truncated normal", the list contains two elements.
        The first element specifies the truncated value and the second element specifies the truncated direction
        (‘left’ or ‘right’). The values at remaining indexes are set to 'NA'. The list must be the same length as
        covnames and in the same order.

    time_thresholds: List, default is None
        A list of integers that splits the time points into different intervals. It is used to create the time variable
        of "categorical time".

    time_points: Int, default is None
        An integer indicating the number of time points to simulate. It is set equal to the maximum number of records
        that obs_data contains for any individual plus 1, if not specified by users.

    n_simul: Int, default is None
        An integer indicating the number of subjects for whom to simulate data. It is set equal to the number of
        subjects in obs_data, if not specified by users.

    baselags: Bool, default is False
        A boolean value specifying the convention used for lagi and lag_cumavgi terms in the model statements when
        pre-baseline times are not included in obs_data and when the current time index, t, is such that t < i. If this
        argument is set to False, the value of all lagi and lag_cumavgi terms in this context are set to 0 (for
        non-categorical covariates) or the reference level (for categorical covariates). If this argument is set to
        True, the value of lagi and lag_cumavgi terms are set to their values at time 0. The default is False.

    visitprocess: List, default is None
        List of lists. Each inner list contains its first entry the covariate name of a visit process; its second entry
        the name of a covariate whose modeling depends on the visit process; and its third entry the maximum number
        of consecutive visits that can be missed before an individual is censored.

    restrictions: List, default is None
        A list with lists, each inner list contains its first entry the covariate name of that its deterministic knowledge
        is known; its second entry is a dictionary whose key is the conditions which should be True when the covariate
        is modeled, the third entry is the value that is set to the covariate during simulation when the conditions
        in the second entry are not True.

    yrestrictions: List, default is None
        A list with lists, for each inner list, its first entry is a dictionary whose key is the conditions which
        should be True when the outcome is modeled, the second entry is the value that is set to the outcome during
        simulation when the conditions in the first entry are not True.

    compevent_restrictions: List, default is None
        A list with lists, for each inner list, its first entry is a dictionary whose key is the conditions which
        should be True when the competing event is modeled, the second entry is the value that is set to the competing
        event during simulation when the conditions in the first entry are not True. Only applicable for survival outcomes.

    basecovs: List, default is None
        A vector of strings specifying the names of baseline covariates in obs_data. These covariates are not
        simulated using a model but keep the same value at all time points. These covariates should not be included
        in covnames.

    parallel: Bool, default is False
        A boolean value indicating whether to parallelize simulations of different interventions to multiple cores.

    ncores: Int, default is None
        An integer indicating the number of cores used in parallelization. It is set to 1 if not specified by users.

    ref_int: Int, default is None
        An integer indicating the intervention to be used as the reference for calculating the end-of-follow-up mean
        ratio and mean difference. 0 denotes the natural course, while subsequent integers denote user-specified
        interventions in the order that they are named in interventions. It is set to 0 if not specified by users.

    ci_method: Str, default is None
        A string specifying the method for calculating the bootstrap 95% confidence intervals, if applicable.
        The options are "percentile" and "normal". It is set to "percentile" if not specified by users.

    seed: Int, default is None
        An integar indicating the starting seed for simulations and bootstrapping. It is set to 1234 if not specified by users.

    save_path: Path, default is None
        A path to save all the returned results. A folder will be created automatically in the current working directory
        if the save_path is not specified by users.

    save_results: Bool, default is False
        A boolean value indicating whether to save all the returned results to the save_path.

    **interventions: Dict, default is None
        A dictionary whose key is the treatment name in the intervention with the format Intervention{id}_{treatment_name},
        value is a list that contains the intervention function, values required by the function, and a list of time
        points in which the intervention is applied.

            """
    def __init__(self,
                 obs_data,
                 id_name,
                 time_name,
                 outcome_name,
                 outcome_model,
                 covnames = None,
                 covtypes = None,
                 covmodels=None,
                 int_descript=None,
                 custom_histvars=None,
                 custom_histories=None,
                 covfits_custom=None,
                 covpredict_custom=None,
                 nsamples=0,
                 compevent_name=None,
                 compevent_model=None,
                 compevent_cens=False,
                 intcomp=None,
                 censor_name=None,
                 censor_model=None,
                 model_fits=False,
                 boot_diag=False,
                 ipw_cutoff_quantile=None,
                 ipw_cutoff_value=None,
                 outcome_type=None,
                 trunc_params=None,
                 time_thresholds=None,
                 time_points=None,
                 n_simul=None,
                 baselags=False,
                 visitprocess=None,
                 restrictions=None,
                 yrestrictions=None,
                 compevent_restrictions=None,
                 basecovs=None,
                 parallel=False,
                 ncores=None,
                 ref_int=None,
                 ci_method=None,
                 seed=None,
                 save_path=None,
                 save_results=False,
                 **interventions
                 ):

        self.obs_data = obs_data
        self.id_name = id_name
        self.time_name = time_name
        self.outcome_name = outcome_name
        self.covnames = covnames
        self.covtypes = covtypes
        self.covmodels = covmodels
        self.outcome_model = outcome_model
        self.int_descript = int_descript
        self.interventions = interventions
        self.custom_histvars = custom_histvars
        self.custom_histories = custom_histories
        self.covfits_custom = covfits_custom
        self.covpredict_custom = covpredict_custom
        self.nsamples = nsamples
        self.compevent_name = compevent_name
        self.compevent_model = compevent_model
        self.compevent_cens = compevent_cens
        self.intcomp = intcomp
        self.censor_name = censor_name
        self.censor_model = censor_model
        self.model_fits = model_fits
        self.boot_diag = boot_diag
        self.ipw_cutoff_quantile = ipw_cutoff_quantile
        self.ipw_cutoff_value = ipw_cutoff_value
        self.outcome_type = outcome_type
        self.trunc_params = trunc_params
        self.time_thresholds = time_thresholds
        self.time_points = time_points
        self.n_simul = n_simul
        self.baselags = baselags
        self.visitprocess = visitprocess
        self.restrictions = restrictions
        self.yrestrictions = yrestrictions
        self.compevent_restrictions = compevent_restrictions
        self.basecovs = basecovs
        self.parallel = parallel
        self.ncores = ncores
        self.ref_int = ref_int
        self.ci_method = ci_method
        self.seed = seed
        self.save_path = save_path
        self.save_results = save_results

        self.set_seed()
        self.origin_obs_data = self.obs_data.copy()

        if self.time_points is None:
            self.time_points = np.max(np.unique(self.obs_data[self.time_name])) + 1

        if self.n_simul is None:
            self.n_simul = len(np.unique(self.obs_data[self.id_name]))

        if self.ncores is None:
            self.ncores = 1

        if self.ci_method is None:
            self.ci_method = 'percentile'

        if self.ref_int is None:
            self.ref_int = 0

        if self.censor_name is not None:
            self.censor = True
        else:
            self.censor = False

        if self.compevent_name is not None and not self.compevent_cens:
            self.competing = True
        else:
            self.competing = False

        if self.visitprocess:
            self.visit_names = [self.visitprocess[i][0] for i in range(len(self.visitprocess))]
            self.visit_covs = [self.visitprocess[i][1] for i in range(len(self.visitprocess))]
            self.max_visits = [self.visitprocess[i][2] for i in range(len(self.visitprocess))]

            self.ts_visit_names = ['ts_' + cov for cov in self.visit_covs]
            for i, cov in enumerate(self.visit_covs):
                self.obs_data = self.obs_data.groupby(self.id_name, group_keys=False).apply(visit_func,
                                         time_name=self.time_name, visit_name=self.visit_names[i],
                                         ts_visit_name=self.ts_visit_names[i])
        else:
            self.visit_names = None
            self.visit_covs = None
            self.max_visits = None
            self.ts_visit_names = None

        if self.covmodels is not None:
            self.cov_hist = get_cov_hist_info(self.covnames, self.covmodels, self.covtypes, self.outcome_model,
                                         self.compevent_model, self.censor_model, self.visit_covs, self.ts_visit_names)
        else:
            self.cov_hist = None

        self.intervention_dicts = read_intervention_input(self.interventions, self.int_descript)


        error_catch(obs_data=self.obs_data, id_name=self.id_name, time_points=self.time_points, time_name=self.time_name,
                    interventions=self.interventions, intervention_dicts=self.intervention_dicts,
                    int_descript=self.int_descript, custom_histvars = self.custom_histvars,
                    custom_histories = self.custom_histories,
                    covfits_custom = self.covfits_custom, covpredict_custom = covpredict_custom,
                    outcome_name=self.outcome_name, covnames=self.covnames, covtypes=self.covtypes,
                    covmodels=self.covmodels, outcome_model=self.outcome_model,
                    compevent_name=self.compevent_name, compevent_model=self.compevent_model,
                    intcomp=self.intcomp, time_thresholds=self.time_thresholds,
                    censor_name=self.censor_name, censor_model=self.censor_model,
                    ipw_cutoff_quantile=self.ipw_cutoff_quantile, ipw_cutoff_value=self.ipw_cutoff_quantile,
                    outcome_type=self.outcome_type, trunc_params=self.trunc_params, basecovs=self.basecovs,
                    ref_int=self.ref_int)

        if self.outcome_type == 'binary_eof' or self.outcome_type == 'continuous_eof':
            self.time_points = np.max(np.unique(self.obs_data[self.time_name])) + 1

        self.int_descript = ['Natural course'] + self.int_descript if self.int_descript is not None else ['Natural course']
        self.intervention_dicts.update({'Natural course': natural})

        if self.covtypes is not None:
            if 'categorical time' in self.covtypes:
                self.obs_data[self.time_name + '_f'] = self.obs_data[time_name].apply(categorical_func,
                                                                                      time_thresholds=self.time_thresholds)
            if 'square time' in self.covtypes:
                self.obs_data['square_' + self.time_name] = self.obs_data[time_name] * self.obs_data[time_name]

        self.below_zero_indicator = (min(self.obs_data[time_name]) < 0)
        if self.covmodels is not None:
            self.update_history()

        if self.outcome_type == 'continuous_eof' or self.outcome_type == 'binary_eof':
            self.intcomp = None
            self.hazardratio = False
            self.competing = False

        if self.outcome_type == 'survival':
            self.hazardratio = True if self.intcomp is not None else False

        self.boot_diag = False if self.nsamples == 0 else self.boot_diag

        if save_results:
            time_now = time.strftime("%Y%m%d_%H-%M-%S", time.localtime())
            if self.save_path is None:
                self.save_path = os.path.join(os.getcwd(), time_now)
                if not os.path.exists(self.save_path):
                    os.makedirs(self.save_path)
            else:
                self.save_path = os.path.join(self.save_path, time_now)
                if not os.path.exists(self.save_path):
                    os.makedirs(self.save_path)
            config_parameters_dict={
                'id_name': self.id_name,
                'time_name': self.time_name,
                'outcome_name': self.outcome_name,
                'covnames': self.covnames,
                'covtypes': self.covtypes,
                'covmodels': self.covmodels,
                'outcome_model': self.outcome_model,
                'int_descript': self.int_descript,
                'nsamples': self.nsamples,
                'competing': self.competing,
                'compevent_name': self.compevent_name,
                'compevent_model': self.compevent_model,
                'compevent_cens': self.compevent_cens,
                'intcomp': self.intcomp,
                'censor': self.censor,
                'censor_name': self.censor_name,
                'censor_model': self.censor_model,
                'model_fits': self.model_fits,
                'boot_diag': self.boot_diag,
                'ipw_cutoff_quantile': self.ipw_cutoff_quantile,
                'ipw_cutoff_value': self.ipw_cutoff_value,
                'outcome_type': self.outcome_type,
                'trunc_params': self.trunc_params,
                'time_thresholds': self.time_thresholds,
                'time_points': self.time_points,
                'n_simul': self.n_simul,
                'baselags': self.baselags,
                'visitprocess': self.visitprocess,
                'basecovs': self.basecovs,
                'parallel': self.parallel,
                'ncores': self.ncores,
                'ref_int': self.ref_int,
                'ci_method': self.ci_method,
                'seed': self.seed
                 }
            save_config(self.save_path, **config_parameters_dict)

    def set_seed(self):
        if self.seed is None:
            self.seed = 1234
        random.seed(self.seed)
        np.random.seed(self.seed)
        # set seeds for simulation and bootstrap sampling
        seeds = random.sample(range(pow(2, 30)), self.nsamples + 1)
        self.simul_seed = seeds[0]
        self.boot_seeds = seeds[1:]

    def update_history(self):
        update_precoded_history(self.obs_data, self.covnames, self.cov_hist, self.covtypes,
                                                             self.time_name, self.id_name, self.below_zero_indicator,
                                                             self.baselags, self.ts_visit_names)
        if self.custom_histvars is not None:
            for t in range(self.time_points):
                update_custom_history(self.obs_data, self.custom_histvars, self.custom_histories,
                                      self.time_name, t, self.id_name)

    def fit(self):

        print('start fitting parametric model.')

        model_coeffs = {}
        model_stderrs = {}
        model_vcovs = {}
        model_fits_summary = {}
        if self.covnames is not None:
            covariate_fits, bounds, rmses, cov_model_coeffs, cov_model_stderrs, cov_model_vcovs, cov_model_fits_summary = \
                fit_covariate_model(covmodels=self.covmodels, covnames=self.covnames, covtypes=self.covtypes,
                                    covfits_custom=self.covfits_custom, time_name=self.time_name, obs_data=self.obs_data,
                                    return_fits=self.model_fits, trunc_params=self.trunc_params, visit_names=self.visit_names,
                                    max_visits=self.max_visits, ts_visit_names=self.ts_visit_names,
                                    visit_covs=self.visit_covs, restrictions=self.restrictions)
            model_coeffs.update(cov_model_coeffs)
            model_stderrs.update(cov_model_stderrs)
            model_vcovs.update(cov_model_vcovs)
            model_fits_summary.update(cov_model_fits_summary)
        else:
            covariate_fits = None
            bounds = None
            rmses = None

        outcome_fit, outcome_model_coeffs, outcome_model_stderrs, outcome_model_vcovs, outcome_model_fits_summary = \
            fit_outcome_model(outcome_model=self.outcome_model, outcome_type=self.outcome_type,
                              outcome_name=self.outcome_name, time_name=self.time_name, obs_data=self.obs_data,
                              competing=self.competing, compevent_name=self.compevent_name, return_fits=self.model_fits,
                              yrestrictions = self.yrestrictions)
        model_coeffs.update(outcome_model_coeffs)
        model_stderrs.update(outcome_model_stderrs)
        model_vcovs.update(outcome_model_vcovs)
        model_fits_summary.update(outcome_model_fits_summary)

        if self.competing:
            compevent_fit, comp_model_coeffs, comp_model_stderrs, comp_model_vcovs, comp_model_fits_summary = \
                fit_compevent_model(compevent_model=self.compevent_model, compevent_name=self.compevent_name,
                                    time_name=self.time_name, obs_data=self.obs_data, return_fits=self.model_fits,
                                    compevent_restrictions=self.compevent_restrictions)
            model_coeffs.update(comp_model_coeffs)
            model_stderrs.update(comp_model_stderrs)
            model_vcovs.update(comp_model_vcovs)
            model_fits_summary.update(comp_model_fits_summary)
        else:
            compevent_fit = None

        if self.censor:
            censor_fit, censor_model_coeffs, censor_model_stderrs, censor_model_vcovs, censor_model_fits_summary = \
                fit_censor_model(censor_model=self.censor_model, censor_name=self.censor_name,
                                 time_name=self.time_name, obs_data=self.obs_data, return_fits=self.model_fits)
            model_coeffs.update(censor_model_coeffs)
            model_stderrs.update(censor_model_stderrs)
            model_vcovs.update(censor_model_vcovs)
            model_fits_summary.update(censor_model_fits_summary)
        else:
            censor_fit = None

        if self.n_simul != len(np.unique(self.obs_data[self.id_name])):
            data_list = dict(list(self.obs_data.groupby(self.id_name, group_keys=True)))
            ids = np.unique(self.obs_data[self.id_name])
            new_ids = np.random.choice(ids, self.n_simul, replace=True)
            new_df = []
            for index, new_id in enumerate(new_ids):
                new_id_df = data_list[new_id].copy()
                new_id_df[self.id_name] = index
                new_df.append(new_id_df)
            data = pd.concat(new_df, ignore_index=True)
        else:
            data = self.obs_data

        print('start simulating.')
        if self.parallel:
            self.all_simulate_results = (
                Parallel(n_jobs=self.ncores)
                (delayed(simulate)(seed=self.simul_seed, time_points=self.time_points, time_name=self.time_name,
                                   id_name=self.id_name, covnames=self.covnames, basecovs=self.basecovs,
                                   covmodels=self.covmodels,  covtypes=self.covtypes, cov_hist=self.cov_hist,
                                   covariate_fits=covariate_fits, rmses=rmses, bounds=bounds, outcome_type=self.outcome_type,
                                   obs_data=data, intervention=self.intervention_dicts[intervention_name],
                                   custom_histvars = self.custom_histvars, custom_histories=self.custom_histories,
                                   covpredict_custom = self.covpredict_custom,
                                   outcome_fit=outcome_fit, outcome_name=self.outcome_name,
                                   competing=self.competing, compevent_name=self.compevent_name,
                                   compevent_fit=compevent_fit, compevent_model=self.compevent_model,
                                   compevent_cens=self.compevent_cens,
                                   trunc_params=self.trunc_params, visit_names=self.visit_names,
                                   visit_covs=self.visit_covs, ts_visit_names=self.ts_visit_names,
                                   max_visits=self.max_visits, time_thresholds=self.time_thresholds,
                                   baselags=self.baselags, below_zero_indicator=self.below_zero_indicator,
                                   restrictions = self.restrictions, yrestrictions=self.yrestrictions,
                                   compevent_restrictions = self.compevent_restrictions)
                 for intervention_name in self.int_descript)
            )
        else:
            self.all_simulate_results = []
            for intervention_name in self.int_descript:
                simulate_result = simulate(seed=self.simul_seed, time_points=self.time_points, time_name=self.time_name,
                                   id_name=self.id_name, covnames=self.covnames, basecovs=self.basecovs,
                                   covmodels=self.covmodels,  covtypes=self.covtypes, cov_hist=self.cov_hist,
                                   covariate_fits=covariate_fits, rmses=rmses, bounds=bounds, outcome_type=self.outcome_type,
                                   obs_data=data, intervention=self.intervention_dicts[intervention_name],
                                   custom_histvars = self.custom_histvars, custom_histories=self.custom_histories,
                                   covpredict_custom = self.covpredict_custom,
                                   outcome_fit=outcome_fit, outcome_name=self.outcome_name,
                                   competing=self.competing, compevent_name=self.compevent_name,
                                   compevent_fit=compevent_fit, compevent_model=self.compevent_model,
                                   compevent_cens=self.compevent_cens,
                                   trunc_params=self.trunc_params, visit_names=self.visit_names,
                                   visit_covs=self.visit_covs, ts_visit_names=self.ts_visit_names,
                                   max_visits=self.max_visits, time_thresholds=self.time_thresholds,
                                   baselags=self.baselags, below_zero_indicator=self.below_zero_indicator,
                                   restrictions=self.restrictions, yrestrictions=self.yrestrictions,
                                   compevent_restrictions=self.compevent_restrictions
                                   )
                self.all_simulate_results.append(simulate_result)

        self.g_results = [res['g_result'] for res in self.all_simulate_results]

        if self.outcome_type == 'survival':
            self.natural_course_risk = self.g_results[0]
        if self.outcome_type == 'binary_eof' or self.outcome_type == 'continuous_eof':
            self.natural_course_risk = None

        self.pools = [res['pool'] for i, res in enumerate(self.all_simulate_results)]
        self.pool_dict = {}
        for i in range(len(self.int_descript)):
            self.pool_dict[self.int_descript[i]] = self.pools[i]
        self.natural_course_pool = self.pool_dict['Natural course']

        # compute non-parametric and parametric covariates means and risks
        self.obs_means, self.est_means, self.obs_res, self.IP_weights = comparison_calculate(
            obs_data=self.obs_data[self.obs_data[self.time_name] >= 0], time_name=self.time_name,
            time_points=self.time_points, id_name=self.id_name, covnames=self.covnames, covtypes=self.covtypes,
            outcome_name=self.outcome_name, outcome_type=self.outcome_type, nc_pool=self.natural_course_pool,
            nc_risk=self.natural_course_risk, competing=self.competing, compevent_name=self.compevent_name,
            compevent_cens=self.compevent_cens, censor=self.censor,
            compevent_fit=compevent_fit, censor_name=self.censor_name,
            censor_fit=censor_fit, ipw_cutoff_quantile=self.ipw_cutoff_quantile,
            ipw_cutoff_value=self.ipw_cutoff_value)

        if self.hazardratio:
            pool1 = self.pools[self.intcomp[0]]
            pool2 = self.pools[self.intcomp[1]]

            if self.competing and not self.compevent_cens:
                import cmprsk.cmprsk as cmprsk

                new_pool1 = pool1.groupby(self.id_name, group_keys=False).apply(hr_comp_data_helper,
                                           outcome_name=self.outcome_name, compevent_name=self.compevent_name)
                new_pool2 = pool2.groupby(self.id_name, group_keys=False).apply(hr_comp_data_helper,
                                           outcome_name=self.outcome_name, compevent_name=self.compevent_name)
                new_pool1['regime'] = 0
                new_pool2['regime'] = 1
                concat_data = pd.concat([new_pool1, new_pool2])
                concat_data = concat_data[[self.time_name, self.outcome_name, self.compevent_name, 'regime']]
                concat_data = concat_data.reset_index(drop=True)
                concat_data['event'] = np.where(concat_data[self.compevent_name] == 1, 2, concat_data[self.outcome_name]).tolist()
                ftime = concat_data[self.time_name]
                fstatus = concat_data['event']
                crr_res = cmprsk.crr(failure_time=ftime, failure_status=fstatus, static_covariates=concat_data[['regime']])
                self.hazard_ratio = crr_res.hazard_ratio()[0][0]
            else:
                new_pool1 = pool1.groupby(self.id_name, group_keys=False).apply(hr_data_helper,
                                                                                outcome_name=self.outcome_name)
                new_pool2 = pool2.groupby(self.id_name, group_keys=False).apply(hr_data_helper,
                                                                                outcome_name=self.outcome_name)
                new_pool1['regime'] = 0
                new_pool2['regime'] = 1
                concat_data = pd.concat([new_pool1, new_pool2])
                concat_data = concat_data[[self.time_name, self.outcome_name, 'regime']]
                cph = CoxPHFitter()
                cph.fit(concat_data, duration_col=self.time_name, event_col=self.outcome_name)
                self.hazard_ratio = cph.hazard_ratios_.values[0]

        if self.nsamples == 0:
            res_table = get_output(ref_int=self.ref_int, int_descript=self.int_descript, censor=self.censor,
                       obs_res=self.obs_res, g_results=self.g_results,  time_points=self.time_points,
                       ci_method=self.ci_method, time_name=self.time_name, obs_means=self.obs_means,
                       outcome_type=self.outcome_type, nsamples=self.nsamples)
            self.boot_table = None

            if self.hazardratio:
                print('Hazardratio value is', '{:.5f}'.format(self.hazard_ratio))

        else:
            if self.parallel:
                boot_results_dicts = (
                    Parallel(n_jobs=self.ncores)
                    (delayed(Bootstrap)(obs_data=self.origin_obs_data, boot_id=i, boot_seeds=self.boot_seeds,
                                                 int_descript=self.int_descript,
                                                 intervention_dicts = self.intervention_dicts,
                                                 covnames=self.covnames, basecovs=self.basecovs, cov_hist=self.cov_hist,
                                                 time_points=self.time_points, n_simul=self.n_simul,
                                                 time_name=self.time_name, id_name=self.id_name,
                                                 custom_histvars=self.custom_histvars, custom_histories=self.custom_histories,
                                                 covpredict_custom=self.covpredict_custom,
                                                 covmodels=self.covmodels, hazardratio=self.hazardratio,
                                                 intcomp=self.intcomp, covtypes=self.covtypes,
                                                 covfits_custom=self.covfits_custom, outcome_model=self.outcome_model,
                                                 outcome_type=self.outcome_type, outcome_name=self.outcome_name,
                                                 competing=self.competing, compevent_name=self.compevent_name,
                                                 compevent_model=self.compevent_model, compevent_cens=self.compevent_cens,
                                                 boot_diag=self.boot_diag, trunc_params=self.trunc_params,
                                                 visit_names=self.visit_names, visit_covs=self.visit_covs,
                                                 ts_visit_names=self.ts_visit_names, max_visits=self.max_visits,
                                                 time_thresholds=self.time_thresholds,
                                                 below_zero_indicator=self.below_zero_indicator, baselags=self.baselags,
                                                 restrictions=self.restrictions, yrestrictions=self.yrestrictions,
                                                 compevent_restrictions=self.compevent_restrictions
                                        )
                     for i in tqdm(range(self.nsamples), desc='Bootstrap progress'))
                )
            else:
                boot_results_dicts = []
                for i in tqdm(range(self.nsamples), desc='Bootstrap progress'):
                    boot_result_dict = Bootstrap(obs_data=self.origin_obs_data, boot_id=i, boot_seeds=self.boot_seeds,
                                                 int_descript=self.int_descript,
                                                 intervention_dicts=self.intervention_dicts,
                                                 covnames=self.covnames, basecovs=self.basecovs, cov_hist=self.cov_hist,
                                                 time_points=self.time_points, n_simul=self.n_simul,
                                                 time_name=self.time_name, id_name=self.id_name,
                                                 custom_histvars=self.custom_histvars, custom_histories=self.custom_histories,
                                                 covpredict_custom=self.covpredict_custom,
                                                 covmodels=self.covmodels, hazardratio=self.hazardratio,
                                                 intcomp=self.intcomp, covtypes=self.covtypes,
                                                 covfits_custom=self.covfits_custom, outcome_model=self.outcome_model,
                                                 outcome_type=self.outcome_type, outcome_name=self.outcome_name,
                                                 competing=self.competing, compevent_name=self.compevent_name,
                                                 compevent_model=self.compevent_model, compevent_cens=self.compevent_cens,
                                                 boot_diag=self.boot_diag, trunc_params=self.trunc_params,
                                                 visit_names=self.visit_names, visit_covs=self.visit_covs,
                                                 ts_visit_names=self.ts_visit_names, max_visits=self.max_visits,
                                                 time_thresholds=self.time_thresholds,
                                                 below_zero_indicator=self.below_zero_indicator, baselags=self.baselags,
                                                 restrictions=self.restrictions, yrestrictions=self.yrestrictions,
                                                 compevent_restrictions=self.compevent_restrictions
                                                 )

                    boot_results_dicts.append(boot_result_dict)

            self.boot_results = [boot_results_dicts[i]['boot_results'] for i in range(self.nsamples) if boot_results_dicts[i]['boot_results'] is not None]

            self.bootests = {'sample_{0}_estimates'.format(i): {self.int_descript[j]:
                             boot_results_dicts[i]['boot_results'][j] for j in range(len(self.int_descript))}
                             for i in range(self.nsamples)}
            self.bootcoeffs = {'sample_{0}_coeffs'.format(i): boot_results_dicts[i]['bootcoeffs'] for i in
                               range(self.nsamples)}
            self.bootstderrs = {'sample_{0}_stderrs'.format(i): boot_results_dicts[i]['bootstderrs'] for i in
                                range(self.nsamples)}
            self.bootvcovs = {'sample_{0}_vcovs'.format(i): boot_results_dicts[i]['bootvcovs'] for i in range(self.nsamples)}

            if self.hazardratio:
                get_hr_output(boot_results_dicts=boot_results_dicts, nsamples=self.nsamples, ci_method=self.ci_method,
                              hazard_ratio=self.hazard_ratio)

            res_table = get_output(ref_int=self.ref_int, int_descript=self.int_descript,
                                   censor=self.censor, obs_res=self.obs_res, g_results=self.g_results,
                                   time_points=self.time_points, ci_method=self.ci_method, time_name=self.time_name,
                                   obs_means=self.obs_means, outcome_type=self.outcome_type, nsamples=self.nsamples,
                                   boot_results=self.boot_results)
            self.boot_table = res_table

        # build results dictionary
        if self.boot_diag:
            self.bootcoeffs = self.bootcoeffs
            self.bootstderrs = self.bootstderrs
            self.bootvcovs = self.bootvcovs
        else:
            self.bootcoeffs = None
            self.bootstderrs = None
            self.bootvcovs = None

        self.summary_dict = {
            'gformula_results': res_table,
            'sim_data': self.pool_dict,
            'IP_weights': self.IP_weights,
            'model_fits_summary': model_fits_summary,
            'model_coeffs': model_coeffs,
            'model_stderrs': model_stderrs,
            'model_vcovs': model_vcovs,
            'rmses': rmses,
            'bounds': bounds,
            'hazard_ratio': '{:.5f}'.format(self.hazard_ratio) if self.hazardratio else 'NA',
            'obs_plot': self.obs_means,
            'est_plot': self.est_means,
            'bootests': None if self.nsamples == 0 else self.bootests,
            'bootcoeffs': self.bootcoeffs,
            'bootstderrs': self.bootstderrs,
            'bootvcovs': self.bootvcovs
        }

        if self.save_results:
            save_results(self.summary_dict, self.save_path)

    def plot_natural_course(self, plot_name='all', colors=None, marker='o', markersize=4, linewidth=0.5,
                            save_figure=False):
        plot_natural_course(time_points=self.time_points, covnames=self.covnames, covtypes=self.covtypes,
                           time_name=self.time_name, obs_data=self.obs_data, obs_means=self.obs_means,
                           est_means=self.est_means, censor=self.censor,
                           outcome_type=self.outcome_type, plot_name=plot_name, colors=colors,
                           marker=marker, markersize=markersize, linewidth=linewidth, save_path=self.save_path,
                           save_figure=save_figure, boot_table=self.boot_table)

    def plot_interventions(self, colors=None, marker='o', markersize=4, linewidth=0.5, save_figure=False):
        plot_interventions(time_points=self.time_points, time_name=self.time_name, risk_results=self.g_results,
                           int_descript=self.int_descript, outcome_type=self.outcome_type,
                           colors=colors, marker=marker, markersize=markersize, linewidth=linewidth,
                           save_path=self.save_path, save_figure=save_figure, boot_table=self.boot_table)