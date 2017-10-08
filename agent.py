import numpy as np
import tensorflow as tf
import gym
from multiprocessing import Lock
from A3C_network import A3CNetwork

# discount factor
GAMMA = 0.99

# number of iterations to store in buffers before updating global net
GLOBAL_UPDATE_INTERVAL = 30

class Agent(object):

    def __init__(self, level_name, worker_name, episode_count):

        # unique worker name
        self.name = worker_name

        # number of total global episodes 
        self.episode_count = episode_count

        # create super mario environment
        self.env = gym.make(level_name)
        #self.env.configure(lock=Lock())
        self.state_n = self.env.observation_space.shape
        self.action_n = self.env.action_space.shape

        # initiate A3C network
        self.a3cnet = A3CNetwork(self.state_n, self.action_n, worker_name)



        """ copy of global network """


    def train(self, sess, coord):
        """ performs the main training loop """

        print(self.name + ' is running')

        # episode loop. Continue playing the game while should not stop
        while not coord.should_stop():

            # reset buffers
            action_buffer, state_buffer, reward_buffer = [], [], []

            # reset game
            s = self.env.reset()

            # reset LSTM memory
            lstm_c = np.zeros((1, self.a3cnet.lstm_cell.state_size.c))
            lstm_h = np.zeros((1, self.a3cnet.lstm_cell.state_size.h))
            # reset LSTM memory for whole batch
            self.batch_lstm_c = lstm_c
            self.batch_lstm_h = lstm_h

            done = False

            # keep track of total reward for entire episode
            total_reward = 0

            # state step loop
            while not done:

                self.env.render()

                policy, value, (lstm_c, lstm_h) = sess.run([
                    self.a3cnet.actor_out,
                    self.a3cnet.critic_out,
                    self.a3cnet.layer4_lstm_state
                ], feed_dict={
                    self.a3cnet.s: [s],
                    self.a3cnet.lstm_c: lstm_c,
                    self.a3cnet.lstm_h: lstm_h
                })
                
                # sample action from the policy distribution at the
                # output of the Actor network
                action = np.zeros(policy.shape[1], dtype=int)
                action[np.random.choice(range(policy.shape[1]), p=policy[0])] = 1

                # take a step in env with action
                s_, r, done, info = self.env.step(action)

                # observe results and store in buffers
                state_buffer.append(s)
                action_buffer.append(action)
                reward_buffer.append(r)
                
                total_reward += r

                if GLOBAL_UPDATE_INTERVAL or done:
                    if done:
                        value_s = 0
                    else:
                        value_s = sess.run(
                            [self.a3cnet.critic_out],
                            feed_dict={
                                self.a3cnet.s: [s],
                                self.a3cnet.lstm_c: lstm_c,
                                self.a3cnet.lstm_h: lstm_h
                            }
                        )
                    # update global net
                    print('value: ' + str(value_s))

                s = s_

            break

        """ TRAIN ONE STEP LOOP """

        """ sample action from network """
        """ perform action, save state & reward to buffer """

        """ if should update global network
            calculate discounted rewards
            update global network 
            pull new copy of global network

        """

        """ if done
          calculate some metrics for tensorboard
          break the TRAIN ONE STEP LOOP to start another episode
        """

if __name__ == '__main__':
    episode_count = tf.Variable(
        0, 
        dtype=tf.int32,
        name='episode_count',
        trainable=False
    )
    agent = Agent('SuperMarioBros-1-1-v0', 'worker_0', episode_count)

    with tf.Session() as sess:
        coord = tf.train.Coordinator()
        sess.run(tf.global_variables_initializer())

        agent.train(sess, coord)
