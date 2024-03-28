import numpy as np
import sys
sys.path.append(r'..\..\pygformula')
from pygformula import ParametricGformula
from pygformula.parametric_gformula.interventions import static
from pygformula.data import load_multiple_treatments_data

obs_data = load_multiple_treatments_data()
time_name = 't0'
id_name = 'id'

covnames = ['L1', 'L2', 'A1', 'A2']
covtypes = ['binary', 'bounded normal', 'binary', 'binary']
covmodels = ['L1 ~ lag1_L1',
             'L2 ~ lag1_L1 + lag1_L2 + lag1_A2 + L1',
             'A1 ~ lag1_L1 + lag1_L2',
             'A2 ~ lag1_A1']

time_points = np.max(np.unique(obs_data[time_name])) + 1
int_descript = ['Always treat on A1 & A2']


outcome_name = 'Y'
outcome_model = 'Y ~ L1 + L2 + A1 + A2'

g = ParametricGformula(obs_data = obs_data, id_name = id_name, time_name=time_name, time_points = time_points,
             covnames=covnames,  covtypes=covtypes, covmodels=covmodels,
             int_descript = int_descript,
             Intervention1_A1 = [static, np.ones(time_points)],
             Intervention1_A2 = [static, np.ones(time_points)],
             outcome_name=outcome_name, outcome_model=outcome_model, outcome_type='survival')
g.fit()