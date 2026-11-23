import numpy as np
from pygformula import ParametricGformula
from pygformula.interventions import static
from pygformula.data import load_continuous_eof

obs_data = load_continuous_eof()
time_name = 't0'
id = 'id'

covnames = ['L1', 'L2', 'A']
covtypes = ['categorical', 'normal', 'binary']
covmodels = ['L1 ~ C(lag1_L1) + lag1_L2 + t0',
             'L2 ~ lag1_L2 + C(lag1_L1) + lag1_A + t0',
              'A ~ C(L1) + L2 + t0']

basecovs = ['L3']

outcome_name = 'Y'
ymodel = 'Y ~ C(L1) + L2 + A'
outcome_type = 'continuous_eof'

time_points = np.max(np.unique(obs_data[time_name])) + 1
int_descript = ['Never treat', 'Always treat']

g = ParametricGformula(obs_data = obs_data, id = id, time_name=time_name,
             int_descript=int_descript,
             Intervention1_A = [static, np.zeros(time_points)],
             Intervention2_A = [static, np.ones(time_points)],
             covnames=covnames, covtypes=covtypes, covmodels=covmodels, basecovs=basecovs,
             outcome_name=outcome_name, ymodel=ymodel, outcome_type=outcome_type
             )
g.fit()
