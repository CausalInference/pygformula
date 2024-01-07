import numpy as np
import sys
sys.path.append(r'..\..\pygformula')
from pygformula import ParametricGformula
from pygformula.parametric_gformula.interventions import static
from pygformula.data import load_absorbing_data

obs_data = load_absorbing_data()
time_name = 't0'
id_name = 'id'

covnames = ['L', 'A']
covtypes = ['absorbing', 'binary']
covmodels = ['L ~ lag1_L + lag1_A + t0',
              'A ~ lag1_A + L + t0']

outcome_name = 'Y'
outcome_model = 'Y ~ L + A + t0'
outcome_type = 'survival'

time_points = np.max(np.unique(obs_data[time_name])) + 1
int_descripts = ['Never treat', 'Always treat']
interventions = [[[static, np.zeros(time_points)]], [[static, np.ones(time_points)]]]
intvars = [['A'], ['A']]

g = ParametricGformula(obs_data = obs_data, id_name = id_name, time_name=time_name, time_points = time_points,
             interventions=interventions, int_descripts = int_descripts, intvars=intvars,
             covnames=covnames, covtypes=covtypes, covmodels=covmodels,
             outcome_name=outcome_name, outcome_model=outcome_model, outcome_type=outcome_type
             )
g.fit()
