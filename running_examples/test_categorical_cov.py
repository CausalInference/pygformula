import numpy as np
import sys
sys.path.append(r'..\..\pygformula')
from pygformula import ParametricGformula
from pygformula.parametric_gformula.interventions import static
from pygformula.data import load_categorical

obs_data = load_categorical()
time_name = 't0'
id = 'id'

covnames = [ 'L', 'A']
covtypes = ['categorical', 'binary']
covmodels = [ 'L ~ C(lag1_L) + t0',
              'A ~ C(L) + C(lag1_L) + t0']

outcome_name = 'Y'
ymodel = 'Y ~ C(lag1_L) + A'

time_points = np.max(np.unique(obs_data[time_name])) + 1
int_descript = ['Never treat', 'Always treat']

g = ParametricGformula(obs_data = obs_data, id = id, time_name=time_name, time_points = time_points,
               int_descript = int_descript,
               Intervention1_A = [static, np.zeros(time_points)],
               Intervention2_A = [static, np.ones(time_points)],
               covnames=covnames,  covtypes=covtypes, covmodels=covmodels, outcome_name=outcome_name,
               ymodel=ymodel, outcome_type='survival')
g.fit()

