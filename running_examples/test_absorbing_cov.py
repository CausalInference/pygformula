import numpy as np
import pygformula
from pygformula import ParametricGformula
from pygformula.interventions import static
from pygformula.data import load_absorbing_data

obs_data = load_absorbing_data()
time_name = 't0'
id = 'id'

covnames = ['L', 'A']
covtypes = ['absorbing', 'binary']
covmodels = ['L ~ lag1_L + lag1_A + t0',
              'A ~ lag1_A + L + t0']

outcome_name = 'Y'
ymodel = 'Y ~ L + A + t0'
outcome_type = 'survival'

time_points = np.max(np.unique(obs_data[time_name])) + 1
int_descript = ['Never treat', 'Always treat']

g = ParametricGformula(obs_data = obs_data, id = id, time_name=time_name, time_points = time_points,
             int_descript = int_descript,
             covnames=covnames, covtypes=covtypes, covmodels=covmodels,
             Intervention1_A = [static, np.zeros(time_points)],
             Intervention2_A = [static, np.ones(time_points)],
             outcome_name=outcome_name, ymodel=ymodel, outcome_type=outcome_type
             )
g.fit()
