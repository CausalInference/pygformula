import numpy as np
import sys
sys.path.append(r'..\..\pygformula')
from pygformula import ParametricGformula
from pygformula.parametric_gformula.interventions import static
from pygformula.data import load_basicdata_nocomp

obs_data = load_basicdata_nocomp()
time_name = 't0'
id_name = 'id'

covnames = ['L2', 'A']
covtypes = ['bounded normal', 'binary']
covmodels = ['L2 ~ lag1_A + lag_cumavg1_L2 + L3 + t0',
           'A ~ lag1_A + L2 + lag_cumavg1_L2 + L3 + t0']

basecovs = ['L3']

outcome_name = 'Y'
outcome_model = 'Y ~ L2 + A + lag1_A + L3 + t0'
outcome_type = 'survival'

time_points = np.max(np.unique(obs_data[time_name])) + 1
int_descripts = ['Never treat', 'Always treat']
interventions = [[[static, np.zeros(time_points)]], [[static, np.ones(time_points)]]]
intvars = [['A'], ['A']]

g = ParametricGformula(obs_data = obs_data, id_name = id_name, time_name=time_name, time_points = time_points,
             interventions=interventions, int_descripts = int_descripts, intvars=intvars,
             covnames=covnames, covtypes=covtypes, covmodels=covmodels, basecovs=basecovs,
             outcome_name=outcome_name, outcome_model=outcome_model, outcome_type=outcome_type
             )
g.fit()