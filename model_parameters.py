#PyPI modules
import matplotlib.pyplot as plt
import seaborn as sns
import csv 
import numpy as np
import pandas as pd
from scipy.io import savemat, loadmat
import os


#loading the time series data
coeff = loadmat("./Power-Converters/RL-buildings/ModelCoeff_Krishna.mat")
Flowrate = pd.read_csv("./Power-Converters/RL-buildings/Flowrates.csv", index_col = 0, parse_dates = True)
Intloads = pd.read_csv("./Power-Converters/RL-buildings/Internalloads.csv", index_col = 0, parse_dates = True)
OthVar = pd.read_csv("./Power-Converters/RL-buildings/OtherVariables.csv", index_col = 0, parse_dates = True)
Temp = pd.read_csv("./Power-Converters/RL-buildings/Temperatures.csv", index_col = 0, parse_dates = True)


#extracting the model coeff.
temp_1, temp_2 = coeff['Coeff'][0][0]
model_params = {
    'a_0': temp_1[0][0],
    'a_1': temp_1[1][0],
    'a_2': temp_1[2][0],
    'a_3': temp_1[3][0],
    'a_4': temp_1[4][0],
    'a_5': temp_1[5][0],
    'b_0': temp_2[0][0],
    'b_1': temp_2[1][0],
    'b_2': temp_2[2][0],
    'b_3': temp_2[3][0],
    'b_4': temp_2[4][0],
    'b_5': temp_2[5][0],
}
for key, value in model_params.items():
    print(key, value)

#parameters
T_set = OthVar.iloc[:, 5]
T_oa = Temp.iloc[:, 14]
Q_int = Intloads.iloc[:, 0]/1000
S_diff = Intloads.iloc[:, 5]
S_dir = Intloads.iloc[:, 6]

inputs = pd.concat([T_set, T_oa, Q_int, S_diff, S_dir], axis = 1)
inputs.index.name = 'Timestamp'
inputs.columns = ['T_set', 'T_oa', 'Q_int', 'S_diff', 'S_dir']

#states
T = Temp.iloc[:,0]
Q = OthVar.iloc[:,12]/1000

states = pd.concat([T,Q], axis = 1)
states.index.name = 'Timestamp'
states.columns = ['T', 'Q']

#extracting june and july data
inputs_e = inputs['2006-06-01' :'2006-07-31']
states_e = states['2006-06-01' :'2006-07-31']
#print(inputs_e)
#print(states_e)