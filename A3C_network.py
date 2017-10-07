import tensorflow as tf
import numpy as np

GLOBAL_SCOPE = 'global'

class A3CNetwork(object):

    def __init__(self, state_n, action_n, scope):
        
        # worker-specific scope / 'global' scope
        with tf.variable_scope(scope):
            # state input placeholder
            self.s = tf.placeholder(tf.float32, shape=[None, 256, 256, 3], name='S')

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
                lstm_cell = tf.contrib.rnn.BasicLSTMCell(256,state_is_tuple=True)

                initial_c = tf.placeholder(tf.float32, [1, lstm_cell.state_size.c])
                initial_h = tf.placeholder(tf.float32, [1, lstm_cell.state_size.h])
                initial_state= tf.contrib.rnn.LSTMStateTuple(
                    initial_c,
                    initial_h
                )

                # TODO: sequence_length
                # Create RNN layer. Also, save lstm state for next training iteration
                self.layer4_lstm_output, self.layer4_lstm_state = tf.nn.dynamic_rnn(
                    lstm_cell, self.layer3_out, initial_state=initial_state,
                    time_major=False
                )

                self.layer4_out = tf.reshape(self.layer4_lstm_output, [-1, 256])

            # define the Actor network (policy)
            with tf.variable_scope('actor'):
                self.actor_layer1 = tf.layers.dense(
                    self.layer4_out,
                    action_n,
                    activation=tf.nn.softmax,
                    name='actor_dense1'
                )

                
            # define the Critic network (value)
            with tf.variable_scope('critic'):
                self.critic_layer1 = tf.layers.dense(
                    self.layer4_out,
                    1,
                    activation=None,
                    name='critic_dense1'
                )

            # only define loss functions for worker networks
            if scope != GLOBAL_SCOPE:
                
                with tf.variable_scope('loss'):
                    pass


    def __new_conv_layer(self, inputs, input_channels_n, filter_size, feature_maps_n, stride, name):
        """ generate a new convolutional layer without any pooling attached

        """
                
        # filter weight shape
        shape = [filter_size, filter_size, input_channels_n, feature_maps_n]
                            
        # new filter weights and biases
        weights = tf.Variable(tf.truncated_normal(shape, stddev=0.05))
        biases = tf.Variable(tf.constant(0.05, shape=[feature_maps_n]))
        
        # new conv layer. stride 1, zero padding VALID
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

    def sync_global_network(self):
        """ use buffers of data from agent to update the global network """
        pass

    def copy_global(self):
        """ extract weights from the global network and copy to agent """
        pass

    def epsilon_greedy_action(self):
        """ perform action with exploration """
        pass

    def action(self):
        """ perform optimal action """
        pass

if __name__ == '__main__':
    tf.reset_default_graph()
    a3cnet = A3CNetwork(256*256, 6, 'global')
    print(a3cnet.layer1_conv)
    print(a3cnet.layer2_conv)
    print(a3cnet.layer2_out)
    print(a3cnet.layer3_dense)
    print(a3cnet.layer4_lstm_state)
    print(a3cnet.layer4_lstm_output)
    print(a3cnet.layer4_out)
    print(a3cnet.actor_layer1)
    print(a3cnet.critic_layer1)
