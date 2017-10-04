import tensorflow as tf
import numpy as np

class A3CNetwork(object):

    def __init__(self, state_n, action_n, scope):
        
        # worker-specific scope / 'global' scope
        with tf.variable_scope(scope):
            # state input placeholder
            self.s = tf.placeholder(tf.float32, shape=[None, 256, 256, 3], name='S')

            # first part of network: CNN
            with tf.variable_scope('CNN'):
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
            with tf.variable_scope('FF1'):
                # get shape for previous layer
                layer2_shape = self.layer2_conv.get_shape()
                # get height * width * feature_maps_n
                features_n = layer2_shape.num_elements()

                # flatten into [None, features_n]
                layer2_flat = tf.reshape(self.layer2_conv, [-1, features_n])
                


    def __new_conv_layer(input, input_channels_n, filter_size, feature_maps_n, stride, name):
        """ generate a new convolutional layer without any pooling attached

        """
                
        # filter weight shape
        shape = [filter_size, filter_size, input_channels_n, feature_maps_n]
                            
        # new filter weights and biases
        weights = tf.Variable(tf.truncated_normal(shape, stddev=0.05))
        biases = tf.Variable(tf.constant(0.05, shape=[num_filters]))
        
        # new conv layer. stride 1, zero padding VALID
        layer = tf.nn.conv2d(
            input=input, 
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
