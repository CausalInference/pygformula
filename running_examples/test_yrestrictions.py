import numpy as np
import sys
sys.path.append(r'..\..\pygformula')
from pygformula import ParametricGformula
from pygformula.parametric_gformula.interventions import static
from pygformula.data import load_basicdata_nocomp

obs_data = load_basicdata_nocomp()

time_name = 't0'
id_name = 'id'

covnames = ['L1', 'L2', 'A']
covtypes = ['binary', 'normal', 'binary']
covmodels = ['L1 ~ lag1_L1 + lag1_A',
             'L2 ~ L1 + lag1_L2',
              'A ~ L1 + L2']

basecovs = ['L3']
outcome_name = 'Y'
outcome_model = 'Y ~ L1 + L2 + A'

# define interventions
time_points = np.max(np.unique(obs_data[time_name])) + 1
int_descripts = ['Never treat', 'Always treat']
interventions = [[[static, np.zeros(time_points)]], [[static, np.ones(time_points)]]]
intvars = [['A'], ['A']]

yrestrictions = [[{'L1': lambda x: x == 0}, 0], [{'L2': lambda x: x > 0.5}, 0.1]]


g = ParametricGformula(obs_data = obs_data, id_name = id_name, time_name=time_name, time_points = time_points,
             interventions=interventions, int_descripts = int_descripts, intvars=intvars,
             covnames=covnames,  covtypes=covtypes, covmodels=covmodels, basecovs=basecovs,
             yrestrictions=yrestrictions, outcome_name=outcome_name, outcome_model=outcome_model, outcome_type='survival')
g.fit()