import numpy as np
import pygformula
from pygformula import ParametricGformula
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

time_points = np.max(np.unique(obs_data[time_name])) + 1

def dynamic_intervention(new_df, pool, int_var, time_name, t):
    new_df.loc[new_df[time_name] == t, int_var] = 0
    new_df.loc[new_df['L2'] > 0.75, int_var] = 1

int_descript = ['Dynamic intervention']

outcome_name = 'Y'
ymodel = 'Y ~ L1 + L2 + L3 + A + lag1_A + lag1_L1 + lag1_L2 + t0'

g = ParametricGformula(obs_data = obs_data, id = id, time_name=time_name, time_points = time_points,
             covnames=covnames,  covtypes=covtypes, covmodels=covmodels, basecovs=basecovs,
             int_descript = int_descript,
             Intervention1_A = [dynamic_intervention],
             outcome_name=outcome_name, ymodel=ymodel, outcome_type='survival')
g.fit()