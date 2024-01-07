import sys
sys.path.append(r'..\..\pygformula')
from pygformula import ParametricGformula
from pygformula.parametric_gformula.interventions import threshold
from pygformula.data import load_binary_eof

obs_data = load_binary_eof()
time_name = 't0'
id_name = 'id'

covnames = ['L1', 'L2', 'A']
covtypes = ['binary', 'zero-inflated normal', 'normal']
covmodels = ['L1 ~ lag1_A + lag2_A + lag_cumavg1_L1 + L3 + t0',
             'L2 ~ lag1_A + L1 + lag_cumavg1_L1 + lag_cumavg1_L2 + L3 + t0',
             'A ~ lag1_A + L1 + L2 + lag_cumavg1_L1 + lag_cumavg1_L2 + L3 + t0']

basecovs = ['L3']

outcome_name = 'Y'
outcome_model = 'Y ~ L1 + A + lag1_A + lag1_L1 + L3 + t0'
outcome_type = 'binary_eof'

int_descripts = ['Threshold intervention']
interventions = [[[threshold, [0.5, float('inf')]]]]
intvars = [['A']]

g = ParametricGformula(obs_data = obs_data, id_name = id_name, time_name=time_name,
             interventions=interventions, int_descripts = int_descripts, intvars=intvars,
             covnames=covnames, covtypes=covtypes, covmodels=covmodels, basecovs=basecovs,
             outcome_name=outcome_name, outcome_model=outcome_model, outcome_type=outcome_type
             )
g.fit()