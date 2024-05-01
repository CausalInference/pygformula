import numpy as np
import pygformula
from pygformula import ParametricGformula
from pygformula.interventions import static
from pygformula.data import load_basicdata_nocomp

obs_data = load_basicdata_nocomp()
time_name = 't0'
id = 'id'

covnames = ['L1', 'L2', 'A', 't0_f']
covtypes = ['binary', 'bounded normal', 'binary', 'categorical time']
covmodels = ['L1 ~ lag1_A + lag2_A + lag_cumavg1_L1 + lag_cumavg1_L2 + L3 + t0 + C(t0_f)',
           'L2 ~ lag1_A + L1 + lag_cumavg1_L1 + lag_cumavg1_L2 + L3 + t0 + C(t0_f)',
           'A ~ lag1_A + L1 + L2 + lag_cumavg1_L1 + lag_cumavg1_L2 + L3 + t0 + C(t0_f)',
           'NA']

time_thresholds = [1, 3, 5]

basecovs = ['L3']

outcome_name = 'Y'
ymodel = 'Y ~ L1 + L2 + L3 + A + lag1_A + lag1_L1 + lag1_L2 + t0'
outcome_type = 'survival'

time_points = np.max(np.unique(obs_data[time_name])) + 1
int_descript = ['Never treat', 'Always treat']

g = ParametricGformula(obs_data = obs_data, id = id, time_name=time_name, time_points = time_points,
             int_descript = int_descript, time_thresholds = time_thresholds,
             Intervention1_A = [static, np.zeros(time_points)],
             Intervention2_A = [static, np.ones(time_points)],
             covnames=covnames, covtypes=covtypes, covmodels=covmodels, basecovs=basecovs,
             outcome_name=outcome_name, ymodel=ymodel, outcome_type=outcome_type,
             )
g.fit()
