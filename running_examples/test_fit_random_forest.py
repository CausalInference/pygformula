import numpy as np
import re
from sklearn.ensemble import RandomForestRegressor

import sys
sys.path.append(r'..\..\pygformula')
from pygformula.parametric_gformula.interventions import static
from pygformula import ParametricGformula
from pygformula.data import load_basicdata_nocomp

obs_data = load_basicdata_nocomp()

time_name = 't0'
id = 'id'

covnames = ['L1', 'L2', 'A']
covtypes = ['binary', 'custom', 'binary']
covmodels = ['L1 ~ lag1_A + lag2_A + lag1_L1 + lag_cumavg1_L2 + t0',
             'L2 ~ lag1_A + L1 + lag1_L1 + lag_cumavg1_L2 + t0',
             'A ~ lag1_A + L1 + L2 +lag1_L1 + lag_cumavg1_L2 + t0']


outcome_name = 'Y'
ymodel = 'Y ~ L1 + L2 + A'

# define interventions
time_points = np.max(np.unique(obs_data[time_name])) + 1
int_descript = ['Never treat', 'Always treat']


def fit_rf(covmodel, covname, fit_data):
    max_depth = 2
    y_name, x_name = re.split('~', covmodel.replace(' ', ''))
    x_name = re.split('\+', x_name.replace(' ', ''))
    y = fit_data[y_name].to_numpy()
    X = fit_data[x_name].to_numpy()
    fit_rf = RandomForestRegressor(max_depth=max_depth, random_state=0)
    fit_rf.fit(X, y)
    return fit_rf

def predict_rf(covmodel, new_df, fit):
    y_name, x_name = re.split('~', covmodel.replace(' ', ''))
    x_name = re.split('\+', x_name.replace(' ', ''))
    X = new_df[x_name].to_numpy()
    prediction = fit.predict(X)
    return prediction

covfits_custom = ['NA', fit_rf, 'NA']
covpredict_custom = ['NA', predict_rf, 'NA']

g = ParametricGformula(obs_data = obs_data, id = id, time_name=time_name, time_points = time_points,
             int_descript = int_descript,
             Intervention1_A = [static, np.zeros(time_points)],
             Intervention2_A = [static, np.ones(time_points)],
             covnames=covnames,  covtypes=covtypes, covmodels=covmodels,
             covfits_custom = covfits_custom, covpredict_custom=covpredict_custom,
             outcome_name=outcome_name, ymodel=ymodel, outcome_type='survival')
g.fit()
