import matplotlib.pyplot as plt
import numpy as np
import gym
from gym import spaces
import matplotlib.pyplot as plt
import argparse
import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import Input, Model, Sequential, layers
import datetime
from scipy.io import savemat
import os
import argparse
import pprint as pp

# local modules
from rl.utils import Scaler, OrnsteinUhlenbeckActionNoise, ReplayBuffer
from rl.gym_env.two_zone_model import two_zone_HVAC
from rl.agents.ddpg import ActorNetwork_rnn, CriticNetwork_rnn, train_rnn

def main(args, reward_result):
    #use GPU
    if args['use_gpu']:
        physical_devices = tf.config.list_physical_devices('GPU') 
        tf.config.experimental.set_memory_growth(physical_devices[0], True)
    path = os.path.dirname(os.path.abspath(__file__)).replace(os.sep, '/')
    A = np.loadtxt(path+'/data/A.txt')
    d = np.loadtxt(path+'/data/d.txt')
    true_states = np.loadtxt(path+'/data/states.txt')
    #initalizing environment
    test_env = two_zone_HVAC(d = d, A = A)
    env = two_zone_HVAC(d = d, A = A)
    print('starting the scaling')
    test_s = test_env.reset()
    test_obs=[]
    test_steps = env.total_no_of_steps-1
    test_episodes = 20
    for _ in range(test_episodes):
        for _ in range(test_steps):
            u = env.action_space.sample()
            s, _,_,_ = test_env.step(u)
            test_obs.append(s)
        test_env.reset()
    scaler = Scaler(2)
    scaler.update(np.concatenate(test_obs).reshape((test_steps*test_episodes,env.observation_space.shape[0])))
    var, mean = scaler.get()
    print('finished the scaling')
    print(var, mean)
    # obs = []
    # for _ in range(200):
    #     s, _, _, _ = test_env.step(0.4)
    #     s_scaled = np.float32((s - mean) * var)
    #     obs.append(s_scaled)
    #plt.plot(np.concatenate(obs).reshape((200,env.observation_space.shape[0])))
    #plt.show()

    ##-----------------------------------------------------------
    state = env.reset()

    actor = ActorNetwork_rnn(
        state_dim = args['state_dim'],
        action_dim = args['action_dim'], 
        action_bound=args['action_bound'], 
        learning_rate = args['actor_lr'], 
        tau = args['tau'], 
        batch_size = args['mini_batch_size'],
        params_rnn = args['actor_rnn'],
        params_l1 = args['actor_l1'],
        params_l2 = args['actor_l2'],
        time_steps = args['time_steps']
        )
    critic = CriticNetwork_rnn(
        state_dim = args['state_dim'], 
        action_dim = args['action_dim'], 
        action_bound = args['action_bound'], 
        learning_rate = args['critic_lr'], 
        tau = args['tau'], 
        gamma = args['gamma'],
        params_rnn = args['critic_rnn'],
        params_l1 = args['critic_l1'],
        params_l2 = args['critic_l2'],
        time_steps = args['time_steps']
        )

    #loading the weights
    if args['load_model']:
        if os.path.isfile(args['summary_dir'] + "/results/actor_weights.h5"):
            print('loading actor weights')
            actor.actor_model.load_weights(args['summary_dir'] + "/results/actor_weights.h5")
        if os.path.isfile(args['summary_dir'] + "/results/target_actor_weights.h5"):
            print('loading actor target weights')
            actor.actor_model.load_weights(args['summary_dir'] + "/results/target_actor_weights.h5")
        if os.path.isfile(args['summary_dir'] + "/results/critic_weights.h5"):
            print('loading critic weights')
            critic.critic_model.load_weights(args['summary_dir'] + "/results/critic_weights.h5")
        if os.path.isfile(args['summary_dir'] + "/results/target_critic_weights.h5"):
            print('loading critic target weights')
            critic.critic_model.load_weights(args['summary_dir'] + "/results/target_critic_weights.h5")

    replay_buffer = ReplayBuffer(int(args['buffer_size']), int(args['random_seed']))
    actor_noise = OrnsteinUhlenbeckActionNoise(mu=np.zeros(action_dim))
    #reward_result = np.zeros(2500)
    print(actor.actor_model.summary())
    print(critic.critic_model.summary())
    paths, reward_result = train_rnn(env, test_env, args, actor, critic, actor_noise, reward_result, scaler, replay_buffer, true_states)

    # Saving the weights
    if args['save_model']:
        actor.actor_model.save_weights(
            filepath = args['summary_dir'] + "/results/actor_weights.h5",
            overwrite=True,
            save_format='h5')
        actor.target_actor_model.save_weights(
            filepath = args['summary_dir'] + "/results/target_actor_weights.h5",
            overwrite=True,
            save_format='h5')
        critic.critic_model.save_weights(
            filepath = args['summary_dir'] + "/results/critic_weights.h5",
            overwrite=True,
            save_format='h5')
        critic.target_critic_model.save_weights(
            filepath = args['summary_dir'] + "/results/target_critic_weights.h5",
            overwrite=True,
            save_format='h5')

    #---------------------------------------------------------------------------
    #Plotting an new testing environment
    if args['scaling']:
        var, mean = scaler.get()
    else:
        var, mean = 1.0, 0.0

    obs_scaled, obs, actions = [], [], []
    test_s = test_env.reset()
    for _ in range(args['time_steps']-1):
        s_scaled = np.float32((test_s - mean) * var)
        obs_scaled.append(s_scaled)
        obs.append(test_s)
        test_s, r, terminal, info = test_env.step(np.array([test_env.action_space.sample()], dtype="float32"))
        actions.append([test_env.action_space.sample()])

    s_scaled = np.float32((test_s - mean) * var)
    obs_scaled.append(s_scaled)
    obs.append(test_s)   
    actions.append([test_env.action_space.sample()])
    for _ in range(test_env.total_no_of_steps-args['time_steps']+1):
        S_0 = obs_scaled[-args['time_steps']:]
        test_a = actor.predict(np.reshape(S_0, (1, args['time_steps'], args['state_dim'])))
        test_s, r, terminal, info = test_env.step(test_a[0])
        s2_scaled = np.float32((test_s - mean) * var)
        obs_scaled.append(s2_scaled)
        if terminal:
            break
    savefig_filename = args['summary_dir']+'/results/results_plot.png'
    #test_env.plot(savefig_filename=savefig_filename)
    test_env.plot(true_states, plot_original=False, savefig_filename = savefig_filename)
    savemat(args['summary_dir'] + '/results/data.mat',
            dict(data=paths, reward=reward_result))

    return [paths, reward_result]

#---------------------------------------------------------------------
if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='provide arguments for DDPG agent')
    #loading the environment to get it default params
    path = os.path.dirname(os.path.abspath(__file__)).replace(os.sep, '/')
    A = np.loadtxt(path+'/data/A.txt')
    d = np.loadtxt(path+'/data/d.txt')
    env = two_zone_HVAC(A=A, d=d)
    state_dim = env.observation_space.shape[0]
    action_dim = env.action_space.shape[0]
    action_bound = env.action_space.high
    #--------------------------------------------------------------------
    parser = argparse.ArgumentParser(description='provide arguments for DDPG agent')

    #general params
    parser.add_argument('--summary_dir', help='directory for saving and loading model and other data', default=path)
    parser.add_argument('--use_gpu', help='weather to use gpu or not', type = bool, default=True)
    parser.add_argument('--save_model', help='Saving model from summary_dir', type = bool, default=True)
    parser.add_argument('--load_model', help='Loading model from summary_dir', type = bool, default=False)
    parser.add_argument('--random_seed', help='seeding the random number generator', default=1754)
    
    #agent params
    parser.add_argument('--buffer_size', help='replay buffer size', type = int, default=1000000)
    parser.add_argument('--max_episodes', help='max number of episodes', type = int, default=200)
    parser.add_argument('--max_episode_len', help='Number of steps per epsiode', type = int, default=env.total_no_of_steps)
    parser.add_argument('--mini_batch_size', help='sampling batch size',type =int, default=300)
    parser.add_argument('--actor_lr', help='actor network learning rate',type =float, default=0.0001)
    parser.add_argument('--critic_lr', help='critic network learning rate',type =float, default=0.001)
    parser.add_argument('--gamma', help='models the long term returns', type =float, default=0.999)
    parser.add_argument('--noise_var', help='Variance of the exploration noise', default=0.0925)
    parser.add_argument('--scaling', help='weather to scale the states before using for training', type = bool, default=True)
    
    #model/env paramerters
    parser.add_argument('--state_dim', help='state dimension of environment', type = int, default=state_dim)
    parser.add_argument('--action_dim', help='action space dimension', type = int, default=action_dim)
    parser.add_argument('--action_bound', help='upper and lower bound of the actions', type = float, default=action_bound)
    parser.add_argument('--discretization_time', help='discretization time used for the environment ', type = float, default=1e-3)

    #Network parameters
    parser.add_argument('--time_steps', help='Number of time-steps for rnn (LSTM)', type = int, default=6)
    parser.add_argument('--actor_rnn', help='actor network rnn paramerters', type = int, default=10)
    parser.add_argument('--actor_l1', help='actor network layer 1 parameters', type = int, default=20)
    parser.add_argument('--actor_l2', help='actor network layer 2 parameters', type = int, default=20)
    parser.add_argument('--critic_rnn', help='critic network rnn parameters', type = int, default=10)
    parser.add_argument('--critic_l1', help='actor network layer 1 parameters', type = int, default=20)
    parser.add_argument('--critic_l2', help='actor network layer 2 parameters', type = int, default=20)
    parser.add_argument('--tau', help='target network learning rate', type = float, default=0.001)
    
    args = vars(parser.parse_args())
    
    pp.pprint(args)

    reward_result = np.zeros(2500)
    [paths, reward_result] = main(args, reward_result)

    savemat('data4_' + datetime.datetime.now().strftime("%y-%m-%d-%H-%M") + '.mat',dict(data=paths, reward=reward_result))

