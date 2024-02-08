import numpy as np
import sys
sys.path.append(r'..\..\pygformula')
from pygformula import ParametricGformula
from pygformula.parametric_gformula.interventions import static
from pygformula.data import load_continuous_eof

obs_data = load_continuous_eof()
time_name = 't0'
id_name = 'id'

covnames = ['L1', 'L2', 'A']
covtypes = ['categorical', 'normal', 'binary']
covmodels = ['L1 ~ C(lag1_L1) + lag1_L2 + t0',
             'L2 ~ lag1_L2 + C(lag1_L1) + lag1_A + t0',
              'A ~ C(L1) + L2 + t0']

basecovs = ['L3']

outcome_name = 'Y'
outcome_model = 'Y ~ C(L1) + L2 + A'
outcome_type = 'continuous_eof'

time_points = np.max(np.unique(obs_data[time_name])) + 1
int_descripts = ['Never treat', 'Always treat']


g = ParametricGformula(obs_data = obs_data, id_name = id_name, time_name=time_name,
             int_descripts=int_descripts,
             Intervention1_A = [static, np.zeros(time_points)],
             Intervention2_A = [static, np.ones(time_points)],
             covnames=covnames, covtypes=covtypes, covmodels=covmodels, basecovs=basecovs,
             outcome_name=outcome_name, outcome_model=outcome_model, outcome_type=outcome_type
             )
g.fit()
