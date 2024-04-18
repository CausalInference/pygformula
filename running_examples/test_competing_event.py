import numpy as np
import pygformula
from pygformula import ParametricGformula
from pygformula.parametric_gformula.interventions import static
from pygformula.data import load_basicdata

obs_data = load_basicdata()

covnames = ['L1', 'L2', 'A']
covtypes = ['binary', 'bounded normal', 'binary']
covmodels = ['L1 ~ lag1_A + lag2_A + lag_cumavg1_L1 + lag_cumavg1_L2 + L3 + t0',
             'L2 ~ lag1_A + L1 + lag_cumavg1_L1 + lag_cumavg1_L2  + L3 + t0',
             'A ~ lag1_A + L1 + L2 +lag_cumavg1_L1 + lag_cumavg1_L2 + L3 + t0']

ymodel = 'Y ~ A + L1 + L2 + L3 + lag1_A + lag1_L1 + lag1_L2'

time_name = 't0'
id = 'id'
outcome_name = 'Y'
basecovs = ['L3']

compevent_name = 'D'
compevent_model = 'D ~ A + L1 + L2 + L3 + t0'

time_points = np.max(np.unique(obs_data[time_name])) + 1
int_descript = ['Never treat', 'Always treat']

g = ParametricGformula(obs_data = obs_data, id = id, time_points = time_points, time_name=time_name,
                  int_descript = int_descript,
                  Intervention1_A = [static, np.zeros(time_points)],
                  Intervention2_A = [static, np.ones(time_points)],
                  basecovs =basecovs, covnames=covnames,  covtypes=covtypes, covmodels=covmodels,
                  compevent_name = compevent_name, compevent_model=compevent_model,
                  outcome_name=outcome_name, outcome_type='survival', ymodel=ymodel)
g.fit()
