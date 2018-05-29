import tensorflow as tf
import numpy as np
import env
import meta_controller_dqn
import controller_dqn

#http://www.davidqiu.com:8888/research/nature14236.pdf      DQN
#https://arxiv.org/pdf/1604.06057.pdf                       HDQN
#https://arxiv.org/pdf/1710.11089.pdf                       Eigenoptions #TODO: Read

environment = env.Env(size = 10, num_obstacles=10, use_key=True)
observation = environment.give_observations()

#TODO: Should the epsilon start at 1.0? What difference will it make?
#TODO: The number of subgoals defined is momentarily static. Is there even a way to make it dynamic? Isnt that the point?

meta_controller = meta_controller_dqn.dqn(environment.objects, num_features=environment.num_features, memory_capacity=50000, target_replacement_rate=200, epsilon=0.9, batch_size=32, decay_rate=0.00001, learning_rate=0.01)
controller = controller_dqn.dqn(environment.num_actions, environment.num_features, memory_capacity=50000, target_replacement_rate= 200, epsilon=0.9, batch_size=32, decay_rate=0.00001, learning_rate=0.01)

k = 4

for episode in range(0, 50000):

    environment.restart()
    step = 0
    terminal = False
    goal_reached = False
    goal = meta_controller.pick_goal(observation)

    while not terminal:

        cumulative_extrinsic_reward = 0
        initial_observation = observation
        while not (terminal or goal_reached):

            action = controller.pick_action(observation, goal)
            previous_observation = observation
            observation, extrinsic_reward, terminal = environment.take_action(action)
            goal_reached, intrinsic_reward = meta_controller.goal_reached(observation, goal)
            controller.store(previous_observation, goal, action, intrinsic_reward, observation, terminal)

            #TODO: Implement different time scales?
            #TODO: Display each epsilon during training
            #TODO: See if the agent learns
            #TODO: Make a context with 1 or more additional keys
            meta_controller.train()
            #if (step > 100 and step % k == 0):
            controller.train()

            cumulative_extrinsic_reward += extrinsic_reward

        meta_controller.store(initial_observation, goal, cumulative_extrinsic_reward, observation, terminal)

        if not terminal:
            goal = meta_controller.pick_goal(observation)
            goal_reached = False

        step += 1

