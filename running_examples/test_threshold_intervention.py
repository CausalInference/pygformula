import numpy as np
import sys
sys.path.append(r'..\..\pygformula')
from pygformula import ParametricGformula
from pygformula.parametric_gformula.interventions import threshold
from pygformula.data import load_threshold_data

obs_data = load_threshold_data()
time_name = 't0'
id_name = 'id'

covnames = ['L1', 'L2', 'A']
covtypes = ['binary', 'bounded normal', 'normal']
covmodels = ['L1 ~ lag1_L1',
             'L2 ~ lag1_L1 + lag1_L2 + L1',
             'A ~ L1 + L2']

time_points = np.max(np.unique(obs_data[time_name])) + 1

int_descripts = ['Threshold intervention']

outcome_name = 'Y'
outcome_model = 'Y ~ L1 + L2 + A'

g = ParametricGformula(obs_data = obs_data, id_name = id_name, time_name=time_name, time_points = time_points,
             covnames=covnames,  covtypes=covtypes, covmodels=covmodels,
             int_descripts = int_descripts,
             Intervention1_A = [threshold, [0.5, float('inf')]],
             outcome_name=outcome_name, outcome_model=outcome_model, outcome_type='survival')
g.fit()