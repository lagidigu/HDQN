import tensorflow as tf
import numpy as np
import env
import env_complex
import meta_controller_dqn
import controller_dqn
import log_tracker

#https://www2.informatik.uni-hamburg.de/wtm/teaching/universitaet/Theses/YounisMSc2015.pdf                              Sohaib Younis Masterarbeit
#http://www.davidqiu.com:8888/research/nature14236.pdf                                                                  DQN
#https://arxiv.org/pdf/1604.06057.pdf                                                                                   HDQN
#https://mindmodeling.org/cogsci2014/papers/221/paper221.pdf                                                            Hierarchical Old
#https://arxiv.org/pdf/1710.11089.pdf                                                                                   Eigenoptions
#https://arxiv.org/pdf/1703.00956.pdf                                                                                   Eigenoptions 2
#https://arxiv.org/pdf/1803.00590.pdf                                                                                   Hierarchical + Imitation Learning combined
#http://www.jneurosci.org/content/38/10/2442                                                                            2018 Neuroscience Related
#https://www.biorxiv.org/content/biorxiv/early/2018/04/13/295964.full.pdf                                               2018 Neuroscience Related (Prefrontal Cortex)
#https://www.ncbi.nlm.nih.gov/pmc/articles/PMC4125626/                                                                  Bio Related (Figure 12A) Ritalin
#www.pnas.org/content/pnas/112/45/13749.full.pdf                                                                        Bio Related model based? Cocaine
#https://arxiv.org/pdf/1802.04765.pdf                                                                                   Transfer learning in the context of HDQN
#https://arxiv.org/pdf/1804.03758.pdf                                                                                   Transfer learning in the context of HDQN
#http://papers.nips.cc/paper/6413-strategic-attentive-writer-for-learning-macro-actions                                 Bridge to planning
#https://arxiv.org/pdf/1511.05952.pdf                                                                                   Prioritized Experience Replay

#TODO: Train only the meta-controller but with binary input information about the keys

iterations = 1000000
meta_controller_training_start = 150000
test_iterations = 100000

log_tracker = log_tracker.log_tracker(num_iterations=iterations)

#environment = env.Env(size=8, num_obstacles=4, use_key=True, logger=log_tracker)
environment = env_complex.EnvComplex(size=10, num_obstacles=8, num_keys=3, logger=log_tracker, draw=False)

meta_controller = meta_controller_dqn.dqn(environment.objects, num_features=environment.num_features,
                                          memory_capacity=50000, target_replacement_rate=50, epsilon=.9,
                                          batch_size=32, decay_rate=1/(0.1 * iterations), learning_rate=0.00025)

controller = controller_dqn.dqn(environment.num_actions, num_features=6, num_goals=len(environment.objects), memory_capacity=1000000,
                                target_replacement_rate=200, epsilon=.9, batch_size=32, decay_rate=1/(0.05 * iterations),
                                learning_rate=0.00025)

time_delay = 0

for episode in range(0, iterations + test_iterations):
    if (episode > iterations):
        time_delay = 0.05
    environment.restart()
    observation = environment.give_observations()
    step = 0
    terminal = False
    goal_reached = False
    goal, goal_num = meta_controller.pick_goal(observation, environment.objects)

    if episode % 5000 == 0:
        print("Current Episode: ", episode)
        #environment.logger.print_logs(episode=episode)
        meta_controller.save()
        controller.save()

    while not terminal:

        cumulative_extrinsic_reward = 0
        initial_observation = observation

        while not (terminal or goal_reached):

            action = controller.pick_action(observation, goal, time_delay)
            previous_observation = observation
            observation, extrinsic_reward, terminal = environment.take_action(action)

            goal_reached, intrinsic_reward = meta_controller.goal_reached(observation, goal)
            controller.store(previous_observation, goal, action, intrinsic_reward, observation, terminal)

            if (episode % 4 == 0):
                meta_controller.train()
                controller.train()

            cumulative_extrinsic_reward += extrinsic_reward

        meta_controller.store(initial_observation, goal, cumulative_extrinsic_reward, observation, terminal)

        if not terminal:
            #if (episode < iterations):
                #environment.logger.log_goal(goal_num, True)
            goal, goal_num = meta_controller.pick_goal(observation, environment.objects)
            goal_reached = False
        else:
            goal_reached = False
            #if (episode < iterations):
                #environment.logger.log_goal(goal_num, goal_reached)
        #if (episode < iterations):
            #environment.logger.log_reward(cumulative_extrinsic_reward, episode)
        step += 1

    if (episode > meta_controller_training_start):
        meta_controller.decay_epsilon(1)
    controller.decay_epsilon(environment.logger.goal_reached_rate, environment.logger.current_epoch)

#environment.logger.plot_cumulative_reward()
