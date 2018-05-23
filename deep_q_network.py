import numpy as np
import tensorflow as tf

class dqn:
    def __init__(self, num_actions, memory_capacity, target_replacement_rate, batch_size, epsilon, learning_rate, discount_factor = 0.99):
        self.num_actions = num_actions
        self.memory_capacity = memory_capacity
        self.target_replacement_rate = target_replacement_rate
        self.batch_size = batch_size
        self.epsilon = epsilon  # TODO: Decay the epsilon
        self.learning_rate = learning_rate

        self.discount_factor = discount_factor
        self.experience_buffer = self.init_experience_buffer()
        self.current_time_step = 0
        self.train_iteration = 0

        self.init_graphs()

        self.session = tf.Session()
        self.session.run(tf.global_variables_initializer())

    def init_experience_buffer(self):
        return np.zeros(self.memory_capacity, 5) #state, action, reward, next_state, terminal

    def init_graphs(self):
        self.define_inputs()

        weights = tf.random_normal_initializer(0.0, 0.3)
        biases = tf.glorot_uniform_initializer(0.1)

        self.build_networks()
        self.update_beginning()
        #Todo: q_eval, loss, train



    def define_inputs(self):
        self.input_state = tf.placeholder(tf.float32, [None, self.num_actions])
        self.input_state_next = tf.placeholder(tf.float32, [None, self.num_actions])
        self.input_reward = tf.placeholder(tf.float32, [None, ])  # TODO: What does the None, mean?
        self.input_action = tf.placeholder(tf.int32, [None, ])

    def build_networks(self):
        with tf.variable_scope("q_network"):
            self.q_network = self.build_q_network(self.input_state)
        with tf.variable_scope("target_network"):
            self.target_network = self.build_q_network(self.input_state_next)


    def build_q_network(self, input_state):
        layer_fc1 = tf.contrib.layers.fully_connected(input_state, 64, activation_fn = tf.nn.relu)
        layer_fc2 = tf.contrib.layers.fully_connected(layer_fc1, self.num_actions, activation_fn = None)
        return layer_fc2

    def update_beginning(self):
        self.update_target_q()
        self.update_q_network()
        self.update_loss()

    def update_target_q(self):
        self.q_target = self.input_reward + self.discount_factor * tf.reduce_max(self.target_network, axis=1)
        self.q_target = tf.stop_gradient(self.q_target)  # No Gradient descent because target network gets updated separately

    def update_q_network(self):
        action_one_hot = tf.one_hot(self.input_action, depth=self.num_actions, dtype=tf.float32)
        self.q_value_for_action = tf.reduce_sum(self.q_network * action_one_hot, axis = 1)

    def update_loss(self):
        self.loss = tf.reduce_mean(tf.squared_difference(self.q_target, self.q_value_for_action)) #TODO: Understand math behind that!
        self.train_operation = tf.train.RMSPropOptimizer(self.learning_rate).minimize(self.loss) #TODO: Understand math behind that!

    def store(self, state, action, reward, next_state, terminal):
        index = self.current_time_step % self.memory_capacity
        experience = [state, action, reward, next_state, terminal]
        self.experience_buffer[index] = experience
        self.current_time_step += 1

    def pick_action(self, state):
        if (np.random.uniform() < self.epsilon):
            action = np.random.randint(0, self.num_actions)
        else:
            action_vals = self.session.run(self.q_network, feed_dict={self.input_state : state})
            action = np.argmax(action_vals)
        return action

    def train(self):
        if (self.train_iteration % self.target_replacement_rate == 0):
            self.reset_target_params()

        if (self.current_time_step > self.memory_capacity):
            indeces = np.random.choice(self.memory_capacity, size=self.batch_size)
        else:
            indeces = np.random.choice(self.current_time_step, size=self.batch_size)
        batch = self.experience_buffer[indeces, :]
        #TODO: Update the network weights

    def reset_target_params(self):
        self.q_network_params = tf.get_collection(tf.GraphKeys.GLOBAL_VARIABLES, scope="q_network")
        self.target_network_params = tf.get_collection(tf.GraphKeys.GLOBAL_VARIABLES, scope="target_network")
        self.session.run([tf.assign(target, q_net) for target, q_net in zip(self.target_network_params, self.q_network_params)])