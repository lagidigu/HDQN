import tensorflow as tf
import numpy as np
import env
import meta_controller_dqn
import controller_dqn
import log_tracker

#http://www.davidqiu.com:8888/research/nature14236.pdf                                                                  DQN
#https://arxiv.org/pdf/1604.06057.pdf                                                                                   HDQN
#https://arxiv.org/pdf/1710.11089.pdf                                                                                   Eigenoptions
#https://arxiv.org/pdf/1703.00956.pdf                                                                                   Eigenoptions 2
#https://arxiv.org/pdf/1803.00590.pdf                                                                                   Hierarchical + Imitation Learning combined
#http://www.jneurosci.org/content/38/10/2442                                                                            2018 Neuroscience Related
#https://www.biorxiv.org/content/biorxiv/early/2018/04/13/295964.full.pdf                                               2018 Neuroscience Related (Prefrontal Cortex)
#https://arxiv.org/pdf/1802.04765.pdf                                                                                   Transfer learning in the context of HDQN
#https://arxiv.org/pdf/1804.03758.pdf                                                                                   Transfer learning in the context of HDQN
#http://papers.nips.cc/paper/6413-strategic-attentive-writer-for-learning-macro-actions                                 Bridge to planning

#TODO: Log Cumulative extrinsic reward for the joint phase.

#TODO: Create an expert class with all the functions given (labelfull, inspectfull, etc.)
#TODO: 


iterations = 200000

log_tracker = log_tracker.log_tracker(num_iterations=iterations)

environment = env.Env(size=10, num_obstacles=10, use_key=True, logger=log_tracker)
observation = environment.give_observations()

meta_controller = meta_controller_dqn.dqn(environment.objects, num_features=environment.num_features,
                                          memory_capacity=50000, target_replacement_rate=1000, epsilon=1,
                                          batch_size=32, decay_rate=1/50000, learning_rate=0.00025)

controller = controller_dqn.dqn(environment.num_actions, environment.num_features, num_goals=len(environment.objects), memory_capacity=1000000,
                                target_replacement_rate=1000, epsilons=1, batch_size=32, decay_rate=1/150000,
                                learning_rate=0.00025)

k = 4
#meta_controller.load()
#controller.load()

for episode in range(0, iterations):

    environment.restart()
    step = 0
    terminal = False
    goal_reached = False
    goal = meta_controller.pick_goal(observation)

    if episode % 1000 == 0:
        environment.logger.print_logs(episode=episode)
        print("Current Episode: ", episode)
        meta_controller.save()
        controller.save()

    while not terminal:

        cumulative_extrinsic_reward = 0
        initial_observation = observation

        while not (terminal or goal_reached):

            action = controller.pick_action(observation, goal)
            previous_observation = observation
            observation, extrinsic_reward, terminal = environment.take_action(action)

            goal_reached, intrinsic_reward = meta_controller.goal_reached(observation, goal)
            controller.store(previous_observation, goal, action, intrinsic_reward, observation, terminal)

            meta_controller.train()
            controller.train()

            cumulative_extrinsic_reward += extrinsic_reward

        meta_controller.store(initial_observation, goal, cumulative_extrinsic_reward, observation, terminal)

        if not terminal:
            environment.logger.log_goal(goal, goal_reached)
            goal = meta_controller.pick_goal(observation)
            goal_reached = False
        else:
            environment.logger.log_goal(goal, goal_reached)

        step += 1

    if (episode > 150000):
        meta_controller.decay_epsilon(1)
    controller.decay_epsilon(environment.logger.goal_reached_rate, environment.logger.current_epoch)



#Fragen
#Wie soll man am besten das annealing abhängig von der success rate gestalten?
#Soll noch viel trainiert werden, nachdem epsilon seinen minimalwert erreicht hat?