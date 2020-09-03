#PyPI modules
import matplotlib.pyplot as plt
import numpy as np
import gym
from gym import spaces

class two_zone_HVAC(gym.Env):
    """
    Buck converter model following gym interface
    We are assuming that the switching frequency is very High
    Action space is continious
    """
    metadata = {'render.modes': ['console']}

    def __init__(self, d, A = None, COP =4):
        super(two_zone_HVAC, self).__init__()

        #parameters
        if A is not None:
            self.A = A
        else:
            self.A = np.array([[0.4670736444788445, 0, 0.473590433381762, 0.027560814480025012, 0.02482360723716469, 0, 0],
                               [0, 0.169849447097808, 1.2326345328482877, -1.2018861561221592, -1.4566448096944626, 0.004739745164037462, 0.002503902132835721]])
        
        #For the default parameters the step size should in the order of 1e-3
        #the steady-state equilibrium of the system is
        self.COP = COP
        self.d=d
        
        #The control action is
        #self.action_space = spaces.Box(low=23, high=26, shape=(1,), dtype=np.float32)
        self.action_space = spaces.Box(low=-1, high=1, shape=(1,), dtype=np.float32)
        self.observation_space = spaces.Box(low=np.array([-np.inf, -np.inf]), high=np.array([+np.inf, +np.inf]), shape=None, dtype=np.float32)
        
        self._get_state()

        #lists to save the states and actions 
        self.state_trajectory = []
        self.action_trajectory = []
        self.count_steps = 0 # counts the number of steps taken

        self.total_no_of_steps = np.shape(self.d)[0]
    
    def _get_state(self):
        #initializing the state vector near to the desired values
        T = np.random.uniform(low = 22, high = 25)
        Q = -np.random.uniform(low = 20, high = 60)
        self.state = np.array([T, Q])

    def _set_state(self, T, Q):
        #using this function we can change the state variable
        self.state = np.array([T, Q])


    def reset(self):
        """
        Important: re-initializing the state vector near to the desired values
        :return: (np.array) 
        """
        self._get_state()
        self.state_trajectory = []
        self.action_trajectory = []
        self.count_steps = 0
        self.state_trajectory.append(self.state)
        return self.state
    
    def step(self, action):
        T_set  =  (action[0] + 1)*(3/2) +23
        T = self.state[0]
        Q = self.state[1]
        state = np.append([T, Q, T_set], self.d[self.count_steps,1:]).T
        self.state = np.matmul(self.A, state)

        # normalize the rewards
        reward = -self.state[1]/self.COP
        
        if self.count_steps+1 == self.total_no_of_steps:
            done = True
        else:
            done = False
        self.count_steps = self.count_steps +  1
        self.state_trajectory.append(self.state)
        self.action_trajectory.append(T_set)
        return self.state, reward, done, {}
    
    def net_state(self):
        if self.count_steps==0:
            print("run env.step before using net_state method")
        return np.append([T, Q], self.d[self.count_steps -1,1:]).T


    def close(self):
        pass

    def plot(self, states, start = 0, end = None, plot_original = True, savefig_filename = None):
        data = np.concatenate(self.state_trajectory).reshape((self.count_steps +1,self.observation_space.shape[0]))
        if end is None:
            end = data.shape[0]-1
        number_of_colors = data.shape[1]
        color = ['r', 'b']
        fig, ax = plt.subplots(nrows=1, ncols=data.shape[1]+1, figsize = (12,4))
        time = np.array(range(data.shape[0]), dtype=np.float32)
        for i in range(data.shape[1]):
            ax[i].plot(time[start:end], data[:, i][start:end],  c = 'b', label='predicted')
            if plot_original:
                ax[i].plot(time[start:end], states[:, i][start:end],c = 'r', marker = '.', label='actual')
            #ax[i].set_ylim(0, des[i]+50)
            ax[i].legend()
        ax[i+1].plot(time[start:end], self.action_trajectory[start:end],  c = 'b', label='Input (T_set)')
        ax[i+1].legend()
        ax[0].set_title('Temperature', fontsize=20)
        ax[0].set_xlabel('Time', fontsize=20)
        ax[1].set_title('Cooling load', fontsize=20)
        ax[1].set_xlabel('Time', fontsize=20)
        if savefig_filename is not None:
            assert isinstance(savefig_filename, str), \
                    "filename for saving the figure must be a string"
            plt.savefig(savefig_filename)
        else:
            plt.show()



