import numpy as np
import re
from sklearn.ensemble import RandomForestRegressor

from pygformula.interventions import static
from pygformula import ParametricGformula
from pygformula.data import load_continuous_eof

obs_data = load_continuous_eof()

time_name = 't0'
id = 'id'

covnames = ['L1', 'L2', 'A']
covtypes = ['categorical', 'normal', 'binary']
covmodels = ['L1 ~ C(lag1_L1) + lag1_L2 + t0',
             'L2 ~ lag1_L2 + C(lag1_L1) + lag1_A + t0',
              'A ~ C(L1) + L2 + t0']

basecovs = ['L3']

outcome_name = 'Y'

ymodel = 'Y ~ lag1_L2 + L2 + lag1_A + A'

# define interventions
time_points = np.max(np.unique(obs_data[time_name])) + 1
int_descript = ['Never treat', 'Always treat']


def ymodel_fit_custom(ymodel, fit_data):
    y_name, x_name = re.split('~', ymodel.replace(' ', ''))
    x_name = re.split('\+', x_name.replace(' ', ''))
    # get feature and target data to fit ymodel
    y = fit_data[y_name].to_numpy()
    X = fit_data[x_name].to_numpy()
    fit_rf = RandomForestRegressor()
    fit_rf.fit(X, y)
    return fit_rf

def ymodel_predict_custom(ymodel, new_df, fit):
    y_name, x_name = re.split('~', ymodel.replace(' ', ''))
    x_name = re.split('\+', x_name.replace(' ', ''))
    # get feature data to predict
    X = new_df[x_name].to_numpy()
    prediction = fit.predict(X)
    return prediction


g = ParametricGformula(obs_data = obs_data, id = id, time_name=time_name, time_points = time_points,
             int_descript = int_descript,
             Intervention1_A = [static, np.zeros(time_points)], basecovs=['L3'],
             Intervention2_A = [static, np.ones(time_points)],
             covnames=covnames,  covtypes=covtypes, covmodels=covmodels,
             ymodel_fit_custom = ymodel_fit_custom, ymodel_predict_custom=ymodel_predict_custom,
             outcome_name=outcome_name, ymodel=ymodel, outcome_type='continuous_eof')
g.fit()