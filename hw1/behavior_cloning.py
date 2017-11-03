import gym
import numpy as np
import tensorflow as tf


# get the env
env = gym.make('Humanoid-v1')

# get the expert data
file_read = np.load('expert_data.npz')
states = file_read['arr_0']
actions = file_read['arr_1']
print(states.shape)
print(actions.shape)

# use the expert data to train a neural net
num_input = env.observation_space.shape[0]
n_hidden_1 = 32
n_hidden_2 = 32
num_output = env.action_space.shape[0]

weights = {}





