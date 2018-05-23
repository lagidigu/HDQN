import tensorflow as tf
import numpy as np

import deep_q_network

#http://www.davidqiu.com:8888/research/nature14236.pdf

observation = "PLACEHOLDER"
env = "PLACEHOLDER"
dqn = deep_q_network.dqn(4, memory_capacity=500, target_replacement_rate= 200, learning_rate=0.01)


for episode in range(0, 1000):
    step = 0
    while True:
        #get the new observation
        action = dqn.pick_action(observation)
        new_observation, reward, terminal = env.take_action(action)
        dqn.store(observation, action, reward, new_observation, terminal)
        dqn.train()


        observation = new_observation

        if (terminal):
            break;

        step += 1

