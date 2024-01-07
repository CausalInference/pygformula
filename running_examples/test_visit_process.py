import numpy as np
import sys
sys.path.append(r'..\..\pygformula')
from pygformula import ParametricGformula
from pygformula.parametric_gformula.interventions import static
from pygformula.data import load_visit_process

obs_data = load_visit_process()
time_name = 'month'
id_name = 'id'

covnames = ['visit_cd4', 'visit_rna', 'cd4_v', 'rna_v', 'everhaart']
covtypes = ['binary', 'binary', 'normal', 'normal', 'binary']
covmodels = ['visit_cd4 ~ lag1_everhaart + lag_cumavg1_cd4_v + sex + race + month',
             'visit_rna ~ lag1_everhaart + lag_cumavg1_rna_v + sex + race + month',
             'cd4_v ~ lag1_everhaart + lag_cumavg1_cd4_v + sex + race + month',
             'rna_v ~ lag1_everhaart + lag_cumavg1_rna_v + sex + race + month',
             'everhaart ~ lag1_everhaart + cd4_v + rna_v + sex + race + month']

basecovs = ['sex', 'race', 'age']

visitprocess = [['visit_cd4', 'cd4_v', 3], ['visit_rna', 'rna_v', 3]]

outcome_name = 'event'
outcome_model = 'event ~ cd4_v + rna_v + everhaart + sex + race + month'

time_points = np.max(np.unique(obs_data[time_name])) + 1

int_descripts = ['Never treat', 'Always treat']
interventions = [[[static, np.zeros(time_points)]], [[static, np.ones(time_points)]]]
intvars = [['everhaart'], ['everhaart']]


g = ParametricGformula(obs_data = obs_data, id_name = id_name,  time_name = time_name, visitprocess = visitprocess,
                  intvars = intvars, interventions = interventions, int_descripts = int_descripts,
                  covnames=covnames,  covtypes=covtypes, covmodels=covmodels, basecovs = basecovs,
                  outcome_name=outcome_name, outcome_model=outcome_model, outcome_type='survival')
g.fit()
g.plot_interventions()
g.plot_natural_course()