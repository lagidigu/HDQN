import numpy as np
import tensorflow as tf
import pickle
import itertools
import time

class dqn:
    def __init__(self, num_actions, num_features, num_goals, memory_capacity, target_replacement_rate, batch_size, epsilon, decay_rate, learning_rate, discount_factor = 0.99):
        self.num_actions = num_actions
        self.num_features = num_features
        self.memory_capacity = memory_capacity
        self.target_replacement_rate = target_replacement_rate
        self.batch_size = batch_size
        self.epsilon = epsilon
        self.decay_rate = decay_rate
        self.learning_rate = learning_rate

        self.discount_factor = discount_factor
        self.init_graphs()
        self.saver = tf.train.Saver()
        self.session = tf.Session()
        try:
            self.experience_buffer = pickle.load(open("controller_experience_buffer.p", "rb"))
            self.current_time_step = pickle.load(open("current_time_step.p", "rb"))
            self.saver.restore(self.session, "tmp/hdqn-controller.ckpt")
            self.epsilon = 0
            print("Controller Model Restored.")
            print("Controller Epsilon is 0.")
        except (OSError, IOError) as e:
            self.session.run(tf.global_variables_initializer())
            self.experience_buffer = np.zeros((self.memory_capacity, self.num_features * 2 + 7))
            self.current_time_step = 1


        self.train_iteration = 0
        self.epsilon_logger = 0

    def init_graphs(self):
        self.define_inputs()
        self.build_networks()
        self.update_beginning()

    def define_inputs(self):
        self.input_state_goal = tf.placeholder(tf.float32, [None, self.num_features + 2]) #last num of features represents goal
        self.input_action = tf.placeholder(tf.int32, [None, ])
        self.input_reward = tf.placeholder(tf.float32, [None, ])
        self.input_state_goal_next = tf.placeholder(tf.float32, [None, self.num_features + 2])
        self.terminal = tf.placeholder(tf.bool, [None, ])

    def build_networks(self):
        with tf.variable_scope("q_network_controller"):
            self.q_network = self.build_q_network(self.input_state_goal)
        with tf.variable_scope("target_network_controller"):
            self.target_network = self.build_q_network(self.input_state_goal_next)

    def build_q_network(self, input_state):
        layer_fc1 = tf.contrib.layers.fully_connected(input_state, 512, activation_fn = tf.nn.relu, scope="fc1")
        #layer_fc2 = tf.contrib.layers.fully_connected(layer_fc1, 128, activation_fn=tf.nn.relu)
        layer_fc3 = tf.contrib.layers.fully_connected(layer_fc1, self.num_actions, activation_fn = None, scope="fc3")
        return layer_fc3

    def update_beginning(self):
        self.update_q_target()
        self.update_q_network()
        self.update_loss()

    def update_q_target(self):
        self.q_target = self.input_reward + self.discount_factor * tf.reduce_max(self.target_network, axis=1)
        self.q_target = tf.stop_gradient(self.q_target)  # No Gradient descent because target network gets updated separately

    def update_q_network(self):
        action_one_hot = tf.one_hot(self.input_action, depth=self.num_actions, dtype=tf.float32)
        self.q_value_for_action = tf.reduce_sum(self.q_network * action_one_hot, axis = 1)

    def update_loss(self):
        self.loss = tf.reduce_mean(tf.squared_difference(self.q_target, self.q_value_for_action))
        self.optimizer = tf.train.RMSPropOptimizer(self.learning_rate).minimize(self.loss)

    def store(self, state, goal, action, reward, next_state, terminal):
        state = state[:self.num_features]
        next_state = next_state[:self.num_features]
        index = self.current_time_step % self.memory_capacity
        experience = list(itertools.chain(state, goal, [action], [reward], next_state, goal, [terminal]))
        self.experience_buffer[index] = experience
        self.current_time_step += 1
        if (self.current_time_step % 100000 == 0):
            pickle.dump(self.experience_buffer, open("controller_experience_buffer.p", "wb"))
            pickle.dump(self.current_time_step, open("current_time_step.p", "wb"))
            print("Controller Experience Buffer Saved, current time step is", self.current_time_step)

    def pick_action(self, state, goal_state, duration):
        state = state[:self.num_features]
        state_goal = np.append(state, goal_state)
        state_goal = state_goal[None, :]
        if (np.random.uniform() < self.epsilon):
            action = np.random.randint(0, self.num_actions)
        else:
            action_vals = self.session.run(self.q_network, feed_dict={self.input_state_goal : state_goal})
            action = np.argmax(action_vals)
        if (duration > 0):
            time.sleep(duration)
        return action

    def train(self):
        if (self.train_iteration % self.target_replacement_rate == 0):
            self.reset_target_params()
        batch = self.get_random_batch()
        _, __ = self.session.run([self.optimizer, self.loss], feed_dict={self.input_state_goal : batch[:, :self.num_features + 2],
                                                                         self.input_action : batch[:, self.num_features + 2],
                                                                         self.input_reward : batch[:, self.num_features + 3],
                                                                         self.input_state_goal_next : batch[:, self.num_features + 4 : -1],
                                                                         self.terminal : batch[:, -1]})
        self.train_iteration += 1

    def get_random_batch(self):
        if (self.current_time_step > self.memory_capacity):
            indeces = np.random.choice(self.memory_capacity, size=self.batch_size)
        else:
            indeces = np.random.choice(self.current_time_step, size=self.batch_size)
        batch = self.experience_buffer[indeces, :]
        return batch

    def decay_epsilon(self, rate, current_epoch):
        if self.epsilon > 0.2:
            self.epsilon -= self.decay_rate
        self.epsilon_logger += 1
        if (self.epsilon_logger % 1000 == 0):
            print("Controller Epsilon:", self.epsilon, ".")

    def reset_target_params(self):
        self.q_network_params = tf.get_collection(tf.GraphKeys.GLOBAL_VARIABLES, scope="q_network_controller")
        self.target_network_params = tf.get_collection(tf.GraphKeys.GLOBAL_VARIABLES, scope="target_network_controller")
        self.session.run([tf.assign(target, q_net) for target, q_net in zip(self.target_network_params, self.q_network_params)])

    def save(self):
        self.save_path = self.saver.save(self.session, "tmp/hdqn-controller.ckpt")
        print("Controller Model Saved.")

