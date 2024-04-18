import numpy as np
import pygformula
from pygformula import ParametricGformula
from pygformula.parametric_gformula.interventions import static
from pygformula.data import load_censor_data

obs_data = load_censor_data()
time_name = 't0'
id = 'id'

covnames = ['L', 'A']
covtypes = ['binary', 'normal']

covmodels = ['L ~ lag1_L + t0',
             'A ~ lag1_A + L + t0']

outcome_name = 'Y'
ymodel = 'Y ~ A + L'

censor_name = 'C'
censor_model = 'C ~ A + L'

time_points = np.max(np.unique(obs_data[time_name])) + 1
int_descript = ['Never treat', 'Always treat']

g = ParametricGformula(obs_data = obs_data, id = id, time_name=time_name, time_points = time_points,
             int_descript=int_descript,
             Intervention1_A = [static, np.zeros(time_points)],
             Intervention2_A = [static, np.ones(time_points)],
             censor_name= censor_name, censor_model=censor_model,
             covnames = covnames, covtypes = covtypes, covmodels = covmodels,
             outcome_name=outcome_name, ymodel=ymodel, outcome_type='survival')
g.fit()
