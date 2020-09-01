#PyPI modules
import matplotlib.pyplot as plt
import seaborn as sns
import csv 
import numpy as np
import pandas as pd
from scipy.io import savemat, loadmat
import os

from rl.gym_env.two_zone_model import two_zone_HVAC
from cbf.cbf import CBF


path = os.path.dirname(os.path.abspath(__file__)).replace(os.sep, '/')

coeff = loadmat(path +"/data/ModelCoeff_Krishna.mat")
Flowrate = pd.read_csv(path +"/data/Flowrates.csv", index_col = 0, parse_dates = True)
Intloads = pd.read_csv(path +"/data/Internalloads.csv", index_col = 0, parse_dates = True)
OthVar = pd.read_csv(path +"/data/OtherVariables.csv", index_col = 0, parse_dates = True)
Temp = pd.read_csv(path +"/data/Temperatures.csv", index_col = 0, parse_dates = True)


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
# for key, value in model_params.items():
#     print(key, value)

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



#Validation and environment testing
A=np.array([[model_params['a_0'],0,model_params['a_1'],model_params['a_2'],model_params['a_3'],model_params['a_4'],model_params['a_5']],
[0,model_params['b_0'],model_params['b_1'],model_params['b_2'],model_params['b_3'],model_params['b_4'],model_params['b_5']]])
d = inputs_e.values
#Saving the A and d arrays
np.savetxt(path+'/data/A.txt', A, fmt='%d')
np.savetxt(path+'/data/d.txt', d, fmt='%d')
env = two_zone_HVAC(d = d, A=A)

Obs, Rew, Done = [], [], []
s = env.reset()
print('sampled action',env.action_space.sample())
Obs.append(s)
done = False
for i in range(10**5):
    if not done:
        s, r, done, _ = env.step([-1.0/3.0])
        Obs.append(s)
        Rew.append(r)
        Done.append(done)
    else:
        break
env.plot(states_e.values, start=0, end=100, plot_original=True, savefig_filename = path+'/results/validation_plot')

#Checking control barrier function based controller
A = np.array([[0.4670736444788445,0, 0.473590433381762, 0.027560814480025012, 0.02482360723716469, 0, 0],
      [0, 0.169849447097808, 1.2326345328482877, -1.2018861561221592, -1.4566448096944626, 0.004739745164037462, 0.002503902132835721]])
d = inputs_e.values
env = two_zone_HVAC(d = d, A=A)
#env._set_state(states_e.values[0][0],states_e.values[0][1])
Obs, Rew, Done = [], [], []
s = env.reset()
Obs.append(s)
done = False
for i in range(10**5):
    if not done:
        
        T_set = CBF(env, T_min = 22, T_max = 23, eta_1 = 0.999, eta_2 = .999)
        action  = (T_set - 23)*(2/3) - 1
        s, r, done, _ = env.step([action])
        Obs.append(s)
        Rew.append(r)
        Done.append(done)
    else:
        break
env.plot(states_e.values, plot_original=False, savefig_filename = path+'/results/cbf_plot')



