import tensorflow as tf
import numpy as np
import env
import meta_controller_dqn
import controller_dqn

#http://www.davidqiu.com:8888/research/nature14236.pdf      DQN
#https://arxiv.org/pdf/1604.06057.pdf                       HDQN

environment = env.Env(size = 10, num_obstacles=10, use_key=False)
observation = environment.give_observations()

#TODO: Is there only 1 feature if the only axis is the subgoal index? Check inside the paper if they give the hyperparameters
#TODO: Should the epsilon start at 1.0? What difference will it make?
#TODO: The number of subgoals defined is momentarily static. Is there even a way to make it dynamic? Isnt that the point?
meta_controller = meta_controller_dqn.dqn(environment.size * environment.size, environment.num_features, memory_capacity=500, target_replacement_rate=200, epsilon=0.9, batch_size=32, decay_rate=0.00001, learning_rate=0.01)
controller = controller_dqn.dqn(environment.num_actions, environment.num_features, memory_capacity=500, target_replacement_rate= 200, epsilon=0.9, batch_size=32, decay_rate=0.00001, learning_rate=0.01)

k = 4


for episode in range(0, 50000):
    environment.restart()
    step = 0
    goal = meta_controller.pick_action(observation)
    while not environment.give_terminal():
        cumulative_extrinsic_reward = 0
        initial_observation = observation
        while not (environment.give_terminal() or meta_controller.goal_reached(observation, goal)):
            action = controller.pick_action()       #TODO: controller needs to pick action based on goal...





        step += 1

