import tensorflow as tf
import numpy as np
import env
import deep_q_network

#http://www.davidqiu.com:8888/research/nature14236.pdf

env = env.Env(size = 5)
observation = env.give_observations()
dqn = deep_q_network.dqn(env.num_actions, env.num_features, memory_capacity=500, target_replacement_rate= 200, epsilon=0.9, batch_size=32, decay_rate=0.0001, learning_rate=0.01)
k = 4

#TODO: Adjust the display to understand what is going on
#TODO: Add punishment for getting out of bounds?

#TODO: Adjust decay rate


for episode in range(0, 1000):
    step = 0
    while True:
        action = dqn.pick_action(observation)

        new_observation, reward, terminal = env.take_action(action)
        dqn.store(observation, action, reward, new_observation, terminal)

        if (step % k == 0):
            dqn.train()

        observation = new_observation

        if (terminal):
            break

        step += 1

