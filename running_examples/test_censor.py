import numpy as np
import sys
sys.path.append(r'..\..\pygformula')
from pygformula import ParametricGformula
from pygformula.parametric_gformula.interventions import static
from pygformula.data import load_censor_data

obs_data = load_censor_data()
time_name = 't0'
id_name = 'id'

covnames = ['L', 'A']
covtypes = ['binary', 'normal']

covmodels = ['L ~ lag1_L + t0',
             'A ~ lag1_A + L + t0']

outcome_name = 'Y'
outcome_model = 'Y ~ A + L'

censor_name = 'C'
censor_model = 'C ~ A + L'

time_points = np.max(np.unique(obs_data[time_name])) + 1
int_descripts = ['Never treat', 'Always treat']
interventions = [[[static, np.zeros(time_points)]], [[static, np.ones(time_points)]]]
intvars = [['A'], ['A']]


g = ParametricGformula(obs_data = obs_data, id_name = id_name, time_name=time_name, time_points = time_points,
             int_descripts=int_descripts, interventions=interventions, intvars=intvars,
             censor_name= censor_name, censor_model=censor_model,
             covnames = covnames, covtypes = covtypes, covmodels = covmodels,
             outcome_name=outcome_name, outcome_model=outcome_model, outcome_type='survival')
g.fit()
