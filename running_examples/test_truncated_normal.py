import numpy as np
import sys
sys.path.append(r'..\..\pygformula')
from pygformula import ParametricGformula
from pygformula.parametric_gformula.interventions import static
from pygformula.data import load_truncated_normal

obs_data = load_truncated_normal()
time_name = 't0'
id_name = 'id'

covnames = ['L', 'A']
covtypes = ['truncated normal', 'binary']
covmodels = ['L ~ lag1_A + lag1_L + t0',
           'A ~ lag1_A + lag1_L + L + t0']

trunc_params = [[1, 'right'], 'NA']

outcome_name = 'Y'
outcome_model = 'Y ~ L + A + t0'
outcome_type = 'survival'

time_points = np.max(np.unique(obs_data[time_name])) + 1
int_descript = ['Never treat', 'Always treat']

g = ParametricGformula(obs_data = obs_data, id_name = id_name, time_name=time_name, time_points = time_points,
             int_descript = int_descript,
             Intervention1_A = [static, np.zeros(time_points)],
             Intervention2_A = [static, np.ones(time_points)],
             covnames=covnames, covtypes=covtypes, covmodels=covmodels, trunc_params=trunc_params,
             outcome_name=outcome_name, outcome_model=outcome_model, outcome_type=outcome_type
             )
g.fit()
