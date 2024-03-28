import numpy as np
import sys
sys.path.append(r'..\..\pygformula')
from pygformula import ParametricGformula
from pygformula.parametric_gformula.interventions import static
from pygformula.data import load_basicdata

obs_data = load_basicdata()

covnames = ['L1', 'L2', 'A']
covtypes = ['binary', 'bounded normal', 'binary']
covmodels = ['L1 ~ lag1_A + lag2_A + lag_cumavg1_L1 + lag_cumavg1_L2 + L3 + t0',
             'L2 ~ lag1_A + L1 + lag_cumavg1_L1 + lag_cumavg1_L2  + L3 + t0',
             'A ~ lag1_A + L1 + L2 +lag_cumavg1_L1 + lag_cumavg1_L2 + L3 + t0']

outcome_model = 'Y ~ A + L1 + L2 + L3 + lag1_A + lag1_L1 + lag1_L2'

time_name = 't0'
id_name = 'id'
outcome_name = 'Y'
basecovs = ['L3']

compevent_name = 'D'
compevent_model = 'D ~ A + L1 + L2 + L3 + t0'
compevent_cens = False

time_points = np.max(np.unique(obs_data[time_name])) + 1
int_descript = ['Never treat', 'Always treat']


compevent_restrictions = [[{'L1': lambda x: x == 0}, 0], [{'L2': lambda x: x > 0.5}, 0.1]]

g = ParametricGformula(obs_data = obs_data, id_name = id_name, time_points = time_points, time_name=time_name,
                  int_descript = int_descript,
                  Intervention1_A = [static, np.zeros(time_points)],
                  Intervention2_A = [static, np.ones(time_points)],
                  basecovs =basecovs, covnames=covnames,  covtypes=covtypes, covmodels=covmodels,
                  compevent_restrictions = compevent_restrictions,
                  compevent_cens= compevent_cens, compevent_name = compevent_name, compevent_model=compevent_model,
                  outcome_name=outcome_name, outcome_type='survival', outcome_model=outcome_model)
g.fit()