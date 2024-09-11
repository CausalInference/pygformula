import numpy as np
from pygformula import ParametricGformula
from pygformula.interventions import static
from pygformula.data import load_basicdata_nocomp

obs_data = load_basicdata_nocomp()

time_name = 't0'
id = 'id'

covnames = ['L1', 'L2', 'A']
covtypes = ['binary', 'normal', 'binary']
covmodels = ['L1 ~ lag1_L1 + lag1_A',
             'L2 ~ L1 + lag1_L2',
              'A ~ L1 + L2']

basecovs = ['L3']
outcome_name = 'Y'
ymodel = 'Y ~ L1 + L2 + A'

# define interventions
time_points = np.max(np.unique(obs_data[time_name])) + 1
int_descript = ['Never treat', 'Always treat']


restrictions = [['L2', {'L1': lambda x: x == 0}, 0.5], ['A', {'L1': lambda x: x == 0, 'L2': lambda x: x > 0.5}, 1]]

g = ParametricGformula(obs_data = obs_data, id = id, time_name=time_name, time_points = time_points,
             int_descript = int_descript,
             Intervention1_A = [static, np.zeros(time_points)],
             Intervention2_A = [static, np.ones(time_points)],
             covnames=covnames,  covtypes=covtypes, covmodels=covmodels, basecovs=basecovs,
             restrictions=restrictions, outcome_name=outcome_name, ymodel=ymodel, outcome_type='survival')
g.fit()
