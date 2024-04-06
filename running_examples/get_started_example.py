import numpy as np
import sys
sys.path.append(r'..\..\pygformula')
from pygformula import ParametricGformula
from pygformula.parametric_gformula.interventions import static
from pygformula.data import load_basicdata_nocomp

obs_data = load_basicdata_nocomp()
time_name = 't0'
id = 'id'

covnames = ['L1', 'L2', 'A']
covtypes = ['binary', 'bounded normal', 'binary']
covmodels = ['L1 ~ lag1_A + lag2_A + lag_cumavg1_L1 + lag_cumavg1_L2 + L3 + t0',
           'L2 ~ lag1_A + L1 + lag_cumavg1_L1 + lag_cumavg1_L2 + L3 + t0',
           'A ~ lag1_A + L1 + L2 + lag_cumavg1_L1 + lag_cumavg1_L2 + L3 + t0']

basecovs = ['L3']

outcome_name = 'Y'
ymodel = 'Y ~ L1 + L2 + L3 + A + lag1_A + lag1_L1 + lag1_L2 + t0'
outcome_type = 'survival'

time_points = np.max(np.unique(obs_data[time_name])) + 1
int_descript = ['Never treat', 'Always treat']


g = ParametricGformula(obs_data = obs_data, id = id, time_name=time_name, time_points = time_points,
             int_descript = int_descript,
             covnames=covnames, covtypes=covtypes, covmodels=covmodels, basecovs=basecovs,
             outcome_name=outcome_name, ymodel=ymodel, outcome_type=outcome_type,
             Intervention1_A = [static, np.zeros(time_points)],
             Intervention2_A = [static, np.ones(time_points)],
             nsamples=20, parallel=True, ncores=8,
             )
g.fit()
g.plot_natural_course()
g.plot_interventions()