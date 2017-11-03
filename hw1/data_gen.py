import load_policy
import gym
import numpy as np
import tensorflow as tf

policy_fn = load_policy.load_policy('./experts/Humanoid-v1.pkl')

env = gym.make('Humanoid-v1')


with tf.Session() as sess:

    states = []
    actions = []

    for _ in range(5):

        state = env.reset()

        for t in range(10000):
            action = policy_fn(state[None, :])

            states.append(state)
            actions.append(action)

            next_state, reward, done, info = env.step(action)
            if done:
                break
            state = next_state


    np.savez('expert_data.npz', states, actions)



