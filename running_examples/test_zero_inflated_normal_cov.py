import numpy as np
import sys
sys.path.append(r'..\..\pygformula')
from pygformula import ParametricGformula
from pygformula.parametric_gformula.interventions import static
from pygformula.data import load_zero_inflated_normal

obs_data = load_zero_inflated_normal()
time_name = 't0'
id = 'id'

covnames = ['L', 'A']
covtypes = ['zero-inflated normal', 'binary']
covmodels = ['L ~ lag1_L + lag1_A + t0',
              'A ~ lag1_A + L + t0']

outcome_name = 'Y'
ymodel = 'Y ~ L + A + t0'
outcome_type = 'survival'

time_points = np.max(np.unique(obs_data[time_name])) + 1
int_descript = ['Never treat', 'Always treat']

g = ParametricGformula(obs_data = obs_data, id = id, time_name=time_name, time_points = time_points,
             int_descript = int_descript,
             Intervention1_A = [static, np.zeros(time_points)],
             Intervention2_A = [static, np.ones(time_points)],
             covnames=covnames, covtypes=covtypes, covmodels=covmodels,
             outcome_name=outcome_name, ymodel=ymodel, outcome_type=outcome_type
             )
g.fit()
