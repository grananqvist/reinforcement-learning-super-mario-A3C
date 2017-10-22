import tensorflow as tf
import numpy as np

GLOBAL_SCOPE = 'global'

# constant for entropy loss influence
BETA = 0.01

class A3CNetwork(object):
    """ The network for policy and value estimation

    There will be one instance of the network for each agent and one additional
    instance called the global network
    """

    def __init__(self, state_shape, action_n, scope):
        
        self.action_n = action_n

        # the scope is 'global' or the agent name
        self.scope = scope

        print('creating network with actions: ' + str(action_n))

        # agent-specific scope / 'global' scope
        with tf.variable_scope(scope):

            # state input placeholder
            state_maps = state_shape[2] if state_shape[2] else 1
            self.s = tf.placeholder(
                tf.float32, 
                shape=[None, state_shape[0], state_shape[1], state_maps],
                name='S'
            )

            # first part of network: CNN
            with tf.variable_scope('cnn'):
                self.layer1_conv = self.__new_conv_layer(
                    self.s,
                    input_channels_n=3,
                    filter_size=8,
                    feature_maps_n=16,
                    stride=4,
                    name='conv1'
                )
                self.layer2_conv = self.__new_conv_layer(
                    self.layer1_conv,
                    input_channels_n=16,
                    filter_size=4,
                    feature_maps_n=32,
                    stride=2,
                    name='conv2'
                )

                # get shape for the last convolution layer
                layer2_shape = self.layer2_conv.get_shape()

                # get height * width * feature_maps_n
                features_n = layer2_shape[1:].num_elements()

                # flatten into [None, features_n]
                self.layer2_out = tf.reshape(self.layer2_conv, [-1, features_n])

            with tf.variable_scope('ff1'):

                # append a fully connected layer on top of convolutions
                self.layer3_dense = tf.layers.dense(
                    self.layer2_out,
                    256,
                    activation=tf.nn.elu,
                    name='dense3',
                )

                # prepare as input for rnn by expanding dims to rank3
                self.layer3_out = tf.expand_dims(self.layer3_dense, [0])

            with tf.variable_scope('rnn'):
                # create lstm cell to use for RNN
                self.lstm_cell = tf.contrib.rnn.BasicLSTMCell(256,state_is_tuple=True)

                # placeholders for the h and c of the LSTM
                self.lstm_c = tf.placeholder(tf.float32, [1, self.lstm_cell.state_size.c])
                self.lstm_h = tf.placeholder(tf.float32, [1, self.lstm_cell.state_size.h])
                initial_state= tf.contrib.rnn.LSTMStateTuple(
                    self.lstm_c,
                    self.lstm_h
                )

                # Create RNN layer. Also, save lstm state for next training iteration
                self.layer4_lstm_output, self.layer4_lstm_state = tf.nn.dynamic_rnn(
                    self.lstm_cell, self.layer3_out, initial_state=initial_state,
                    time_major=False
                )

                self.layer4_out = tf.reshape(self.layer4_lstm_output, [-1, 256])

            # define the Actor network (policy)
            with tf.variable_scope('actor'):

                # currently implemented using only one fully connected layer
                # output is a distribution of action probabilities
                self.actor_out = tf.layers.dense(
                    self.layer4_out,
                    action_n,
                    activation=tf.nn.softmax,
                    name='actor_dense1'
                )

                
            # define the Critic network (value)
            with tf.variable_scope('critic'):
                # currently also implemented using only one fully connected layer
                # output is an estimation of the value function V(s)
                self.critic_out = tf.layers.dense(
                    self.layer4_out,
                    1,
                    activation=None,
                    name='critic_dense1'
                )

            # only define loss functions for agent networks
            if scope != GLOBAL_SCOPE:
                
                with tf.variable_scope('loss'):

                    # the batch of rewards 
                    self.reward = tf.placeholder(tf.float32, shape=[None, 1], name='R')

                    # the batch of advantages 
                    self.advantage = tf.placeholder(tf.float32, shape=[None, 1], name='A')

                    # the action taken
                    self.action_taken = tf.placeholder(tf.float32, shape=[None, self.action_n], name='Action')

                    # the log probability of taking action i given state. log(pi(a_i|s_i))
                    # The probability is extracted by multiplying by the one hot action vector
                    # add small number (1e-10) to prevent NaN
                    logp = tf.log(tf.reduce_sum(
                        self.actor_out * self.action_taken, axis=1, keep_dims=True
                    ) + 1e-10)

                    # calculate H(pi), the policy entropy
                    # Add small number here aswell for the same reason as for logp
                    self.policy_entropy = -1 * BETA * tf.reduce_sum(self.actor_out * tf.log(self.actor_out + 1e-10))

                    # the actor loss, policy gradient
                    self.actor_loss = -1 * tf.reduce_sum(logp * self.advantage) 

                    # loss function for Critic network
                    # multiplied by 1/2 to reduce its influence for backprop for more stable learning
                    self.critic_loss = 1/2 * tf.nn.l2_loss(self.reward - self.critic_out)

                    # combine into one optimizable loss function
                    self.loss = self.critic_loss + self.actor_loss - self.policy_entropy

                    # init an optimizer to use when training
                    self.optimizer = tf.train.AdamOptimizer(learning_rate=1e-4)

                    # init operation for updating global network
                    self._init_sync_global_network()

                    # init operation for syncing this network with the global network
                    self._init_copy_global_network()

                    # summaries for logging performance metrics to tensorboard

                    tf.summary.scalar(
                        'loss', 
                        tf.reduce_mean(self.loss), 
                        collections=[self.scope]
                    )
                    tf.summary.scalar(
                        'policy_entropy', 
                        self.policy_entropy,
                        collections=[self.scope]
                    )
                    tf.summary.scalar(
                        'actor_loss', 
                        self.actor_loss,
                        collections=[self.scope]
                    )
                    tf.summary.scalar(
                        'critic_loss', 
                        self.critic_loss,
                        collections=[self.scope]
                    )

                 
            self.weights_summary_op = tf.summary.merge_all(GLOBAL_SCOPE)
            self.summary_op = tf.summary.merge_all(self.scope)


    def __new_conv_layer(self, inputs, input_channels_n, filter_size, feature_maps_n, stride, name):
        """ generate a new convolutional layer without any pooling attached """
                
        # filter weight shape
        shape = [filter_size, filter_size, input_channels_n, feature_maps_n]
                            
        # new filter weights and biases
        weights = tf.Variable(tf.truncated_normal(shape, stddev=0.05))
        biases = tf.Variable(tf.constant(0.05, shape=[feature_maps_n]))

        if self.scope == GLOBAL_SCOPE:
            # keep summary of weights for the layers if global network, used for debugging
            tf.summary.histogram(name + '/W/histogram', weights, collections=[self.scope])
            tf.summary.histogram(name + '/b/histogram', biases, collections=[self.scope])
        
        # new conv layer. using above configuration. zero padding VALID
        layer = tf.nn.conv2d(
            input=inputs, 
            filter=weights, 
            strides=[1,stride,stride,1], 
            padding='VALID',
            name=name
        )
                                                        
        # add biases to each feature map
        layer += biases
                                                                                            
        # pass layer through an ELU
        layer = tf.nn.elu(layer)

        return layer

    def _init_sync_global_network(self):
        """ use buffers of data from agent to update the global network """

        # get trainable variables for agent
        agent_weights = tf.get_collection(tf.GraphKeys.TRAINABLE_VARIABLES, self.scope)
        
        # get the gradients of the agent's trainable variables
        agent_gradients = tf.gradients(self.loss, agent_weights)

        # get trainable variables for global network
        global_weights = tf.get_collection(tf.GraphKeys.TRAINABLE_VARIABLES, GLOBAL_SCOPE)

        # clip gradients because they can get too big for large magnitude of rewards
        clipped_gradients, _ = tf.clip_by_global_norm(agent_gradients,40.0)

        # minimize loss by applying gradients to global network
        self.sync_global_network = self.optimizer.apply_gradients(zip(clipped_gradients, global_weights))


    def _init_copy_global_network(self):
        """ extract weights from the global network and copy to agent """

        # get trainable variables for agent
        agent_weights = tf.get_collection(tf.GraphKeys.TRAINABLE_VARIABLES, self.scope)

        # get trainable variables for global network
        global_weights = tf.get_collection(tf.GraphKeys.TRAINABLE_VARIABLES, GLOBAL_SCOPE)

        # loop over weights and copy the global over to the agent
        # return the operations
        print('copying global')
        self.copy_global_network = [ agent_w.assign(global_w) for agent_w, global_w 
                in zip(agent_weights, global_weights)]
        

    def epsilon_greedy_action(self, sess, s, epsilon):
        """ perform action with exploration 
        
        NOTE! CURRENTLY NOT NEEDED BECAUSE OF THE STOCHASTICITY OF
        THE POLICY NETWORK OUTPUT
        """

        if np.random.uniform() < epsilon:
            action = np.zeros(self.action_n,1)
            action[np.random.randint(0, len(self.action_n))] = 1
        else:
            action = self.sample_action(s)

        return action

    def sample_action(self, sess, s):
        """ not needed to implement """
        pass
