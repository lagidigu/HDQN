import numpy as np
import tensorflow as tf

class dqn:
    def __init__(self, num_actions, num_features, memory_capacity, target_replacement_rate, batch_size, epsilon, decay_rate, learning_rate, discount_factor = 0.99):
        self.num_actions = num_actions
        self.num_features = num_features
        self.memory_capacity = memory_capacity
        self.target_replacement_rate = target_replacement_rate
        self.batch_size = batch_size
        self.epsilon = epsilon
        self.decay_rate = decay_rate
        self.learning_rate = learning_rate

        self.discount_factor = discount_factor

        self.experience_buffer = np.zeros((self.memory_capacity, self.num_features * 2 + 5))

        self.current_time_step = 1
        self.train_iteration = 0

        self.init_graphs()

        self.session = tf.Session()
        self.session.run(tf.global_variables_initializer())

    def init_graphs(self):
        self.define_inputs()
        self.build_networks()
        self.update_beginning()

    def define_inputs(self):
        self.input_state_goal = tf.placeholder(tf.float32, [None, self.num_features + 1]) #last num of features represents goal
        self.input_action = tf.placeholder(tf.int32, [None, ])
        self.input_reward = tf.placeholder(tf.float32, [None, ])
        self.input_state_goal_next = tf.placeholder(tf.float32, [None, self.num_features + 1])
        self.terminal = tf.placeholder(tf.bool, [None, ])

    def build_networks(self):
        with tf.variable_scope("q_network_controller"):
            self.q_network = self.build_q_network(self.input_state_goal)
        with tf.variable_scope("target_network_controller"):
            self.target_network = self.build_q_network(self.input_state_goal_next)

    def build_q_network(self, input_state):
        layer_fc1 = tf.contrib.layers.fully_connected(input_state, 64, activation_fn = tf.nn.relu)
        layer_fc2 = tf.contrib.layers.fully_connected(layer_fc1, self.num_actions, activation_fn = None)
        return layer_fc2

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
        index = self.current_time_step % self.memory_capacity
        experience = [state[0], state[1], goal, action, reward,
                      next_state[0], next_state[1], goal, terminal] #TODO: Not n_feature dynamic!
        self.experience_buffer[index] = experience
        self.current_time_step += 1

    def pick_action(self, state, goal):
        state_goal = np.append(state, goal)
        state_goal = state_goal[None, :]
        if (np.random.uniform() < self.epsilon):
            action = np.random.randint(0, self.num_actions)
        else:
            action_vals = self.session.run(self.q_network, feed_dict={self.input_state_goal : state_goal})
            action = np.argmax(action_vals)
        return action

    def train(self):
        if (self.train_iteration % self.target_replacement_rate == 0):
            self.reset_target_params()

        batch = self.get_random_batch()

        _, __ = self.session.run([self.optimizer, self.loss], feed_dict={self.input_state_goal : batch[:, :self.num_features + 1],
                                                                         self.input_action : batch[:, self.num_features + 1],
                                                                         self.input_reward : batch[:, self.num_features + 2],
                                                                         self.input_state_goal_next : batch[:, self.num_features + 3 : -1],
                                                                         self.terminal : batch[:, -1]})

        self.train_iteration += 1
        self.decay_epsilon()

    def get_random_batch(self):
        if (self.current_time_step > self.memory_capacity):
            indeces = np.random.choice(self.memory_capacity, size=self.batch_size)
        else:
            indeces = np.random.choice(self.current_time_step, size=self.batch_size)
        batch = self.experience_buffer[indeces, :]
        return batch

    def decay_epsilon(self):
        if (self.epsilon > 0.1):
            self.epsilon -= self.decay_rate
            #if (str(self.epsilon)[3] == "0"):
                #print("Epsilon: ", self.epsilon)

    def reset_target_params(self):
        self.q_network_params = tf.get_collection(tf.GraphKeys.GLOBAL_VARIABLES, scope="q_network_controller")
        self.target_network_params = tf.get_collection(tf.GraphKeys.GLOBAL_VARIABLES, scope="target_network_controller")
        self.session.run([tf.assign(target, q_net) for target, q_net in zip(self.target_network_params, self.q_network_params)])