import numpy as np
import tensorflow as tf
import pickle

class dqn:
    def __init__(self, objects, num_features, memory_capacity, target_replacement_rate, batch_size, epsilon, decay_rate, learning_rate, discount_factor = 0.99):
        #self.goals_shape = goals_shape

        # This will be the goal space.
        # If we can imagine every goal from G as {Entity A, Relation, Entity B},
        # We can make the generalization that A is the Agent, relations is "is near",
        # and Entity B is the Object in question. This can be extended by :
        #
        # Changing Entity A to another object than object A.
        # Changing Relation to continuous functions such as "Distance".
        # Creating Meta-Goals involving multiple Relations between Entities.

        self.objects = objects

        self.num_features = num_features
        self.num_possible_goals = 0#int(np.prod(self.goals_shape))

        self.memory_capacity = memory_capacity
        self.target_replacement_rate = target_replacement_rate
        self.batch_size = batch_size
        self.epsilon = epsilon
        self.epsilon_logger = 0
        self.decay_rate = decay_rate
        self.learning_rate = learning_rate

        self.discount_factor = discount_factor

        try:
            self.experience_buffer = pickle.load(open("meta_controller_experience_buffer.p", "rb"))
            self.current_time_step = pickle.load(open("meta_current_time_step.p", "rb"))
        except (OSError, IOError) as e:
            self.experience_buffer = np.zeros((self.memory_capacity, self.num_features * 2 + 3))
            self.current_time_step = 1

        self.train_iteration = 0

        self.init_graphs()

        self.session = tf.Session()
        self.session.run(tf.global_variables_initializer())

        self.saver = tf.train.Saver()


    def init_graphs(self):
        self.define_inputs()
        self.build_networks()
        self.update_beginning()

    def define_inputs(self):
        self.input_state = tf.placeholder(tf.float32, [None, self.num_features], name="meta_input_state")
        self.input_goal = tf.placeholder(tf.int32, [None, ], name="meta_input_goal")
        self.input_extrinsic_reward = tf.placeholder(tf.float32, [None, ], name="meta_input_reward")
        self.input_state_next = tf.placeholder(tf.float32, [None, self.num_features], name="meta_input_state_next")
        self.terminal = tf.placeholder(tf.bool, [None, ], name="meta_input_terminal")

    def build_networks(self):
        with tf.variable_scope("q_network_meta_controller"):
            self.q_network = self.build_q_network(self.input_state)
        with tf.variable_scope("target_network_meta_controller"):
            self.target_network = self.build_q_network(self.input_state_next)

    #TODO: Fine-Tune Neuron Amount.
    def build_q_network(self, input_state):
        layer_fc1 = tf.contrib.layers.fully_connected(input_state, 128, activation_fn = tf.nn.relu)
        layer_fc2 = tf.contrib.layers.fully_connected(layer_fc1, len(self.objects), activation_fn = None)
        return layer_fc2

    def update_beginning(self):
        self.update_q_target()
        self.update_q_network()
        self.update_loss()

    def update_q_target(self):
        self.q_target = self.input_extrinsic_reward + self.discount_factor * tf.reduce_max(self.target_network, axis=1)
        self.q_target = tf.stop_gradient(self.q_target)  # No Gradient descent because target network gets updated separately

    def update_q_network(self):
        goal_one_hot = tf.one_hot(self.input_goal, depth=len(self.objects), dtype=tf.float32)
        self.q_value_for_goal = tf.reduce_sum(self.q_network * goal_one_hot, axis = 1)

    def update_loss(self):
        self.loss = tf.reduce_mean(tf.squared_difference(self.q_target, self.q_value_for_goal))
        self.optimizer = tf.train.RMSPropOptimizer(self.learning_rate).minimize(self.loss)

    def store(self, state, goal, extrinsic_reward, next_state, terminal):
        index = self.current_time_step % self.memory_capacity
        experience = [state[0], state[1], goal, extrinsic_reward, next_state[0], next_state[1], terminal] #TODO: Not n_feature dynamic!
        self.experience_buffer[index] = experience
        self.current_time_step += 1
        if (self.current_time_step % 25000 == 0):
            pickle.dump(self.experience_buffer, open("meta_controller_experience_buffer.p", "wb"))
            pickle.dump(self.current_time_step, open("meta_current_time_step.p", "wb"))
            print("Meta-Controller Experience Buffer Saved, current time step is", self.current_time_step)


    def pick_goal(self, state):
        if (np.random.uniform() < self.epsilon):
            # goal = np.zeros(len(self.goals_shape), dtype=float)
            # for i in self.goals_shape:
            #     num = np.random.randint(0, i)
            #     goal[self.goals_shape.index(i)] = num
            goal = np.random.randint(0, len(self.objects))
        else:
            state = state[None, :]
            goal_vals = self.session.run(self.q_network, feed_dict={self.input_state : state})
            goal = np.argmax(goal_vals)
        return goal

    def train(self):
        if (self.train_iteration % self.target_replacement_rate == 0):
            self.reset_target_params()

        batch = self.get_random_batch()

        _, __ = self.session.run([self.optimizer, self.loss], feed_dict={self.input_state : batch[:, :self.num_features],
                                                                         self.input_goal : batch[: , self.num_features],
                                                                         self.input_extrinsic_reward : batch[:, self.num_features + 1],
                                                                         self.input_state_next : batch[:, self.num_features + 2 : -1],
                                                                         self.terminal : batch[:, -1]})

        self.train_iteration += 1
        #self.decay_epsilon() #TODO: Decay based on the average success rate of reaching goal g?

    # TODO: Generalize? Paper hasnt done that yet.
    def goal_reached(self, observation, goal):
        reached = False
        intrinsic_reward = 0
        comparison = observation == self.objects[goal]
        if (np.array(comparison).all()):
            reached = True
            intrinsic_reward = 1 #TODO: You could base the intrinsic reward on the distance
        else:
            intrinsic_reward = -1
        return reached, intrinsic_reward

    def get_random_batch(self):
        if (self.current_time_step > self.memory_capacity):
            indeces = np.random.choice(self.memory_capacity, size=self.batch_size)
        else:
            indeces = np.random.choice(self.current_time_step, size=self.batch_size)
        batch = self.experience_buffer[indeces, :]
        return batch

    def decay_epsilon(self, rate):
        if (self.epsilon > 0.01):
            self.epsilon -= self.decay_rate * rate
            self.epsilon_logger += 1
            if (self.epsilon_logger % 1000 == 0):
                print("Epsilon Value : ", self.epsilon)

    def reset_target_params(self):
        self.q_network_params = tf.get_collection(tf.GraphKeys.GLOBAL_VARIABLES, scope="q_network_meta_controller")
        self.target_network_params = tf.get_collection(tf.GraphKeys.GLOBAL_VARIABLES, scope="target_network_meta_controller")
        self.session.run([tf.assign(target, q_net) for target, q_net in zip(self.target_network_params, self.q_network_params)])

    def save(self):
        self.save_path = self.saver.save(self.session, "tmp/hdqn.ckpt")
        print("Meta Model Saved.")

    def load(self):
        self.saver.restore(self.session, "tmp/hdqn.ckpt")
        print("Meta Model Restored.")
