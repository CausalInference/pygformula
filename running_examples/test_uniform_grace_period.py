import numpy as np
import sys
sys.path.append(r'..\..\pygformula')
from pygformula import ParametricGformula
from pygformula.parametric_gformula.interventions import uniform_grace_period
from pygformula.data import load_basicdata_nocomp

obs_data = load_basicdata_nocomp()
time_name = 't0'
id_name = 'id'

covnames = ['L1', 'L2', 'A']
covtypes = ['binary', 'bounded normal', 'binary']
covmodels = ['L1 ~ lag1_A + lag2_A + lag_cumavg1_L1 + lag_cumavg1_L2 + L3 + t0',
             'L2 ~ lag1_A + L1 + lag_cumavg1_L1 + lag_cumavg1_L2 + L3 + t0',
             'A ~ lag1_A + L1 + L2 + lag_cumavg1_L1 + lag_cumavg1_L2 + L3 + t0']

basecovs = ['L3']

time_points = np.max(np.unique(obs_data[time_name])) + 1

int_descripts = ['uniform grace period intervention']
conditions = {'L1': lambda x: x == 1}
interventions = [[[uniform_grace_period, [3, conditions]]]]
intvars = [['A']]

outcome_name = 'Y'
outcome_model = 'Y ~ L1 + L2 + L3 + A + lag1_A + lag1_L1 + lag1_L2 + t0'

g = ParametricGformula(obs_data = obs_data, id_name = id_name, time_name=time_name, time_points = time_points,
             covnames=covnames,  covtypes=covtypes, covmodels=covmodels, basecovs=basecovs,
             int_descripts = int_descripts, interventions = interventions, intvars = intvars,
             outcome_name=outcome_name, outcome_model=outcome_model, outcome_type='survival')
g.fit()