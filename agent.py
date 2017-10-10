import numpy as np
import tensorflow as tf
import gym
from time import sleep
from multiprocessing import Lock
from A3C_network import A3CNetwork

# discount factor
GAMMA = 0.99

# number of iterations to store in buffers before updating global net
GLOBAL_UPDATE_INTERVAL = 30

class Agent(object):

    def __init__(self, level_name,agent_name, episode_count):

        # unique agent name
        self.name = agent_name

        # number of total global episodes 
        self.episode_count = episode_count

        # create super mario environment
        self.env = gym.make(level_name)
        #self.env.configure(lock=Lock())
        self.state_n = self.env.observation_space.shape
        self.action_n = self.env.action_space.shape

        # initiate A3C network
        self.a3cnet = A3CNetwork(self.state_n, self.action_n, agent_name)

    def train(self, sess, coord):
        """ performs the main training loop """

        print(self.name + ' is running')

        step_counter = 1

        s = self.env.reset()

        # episode loop. Continue playing the game while should not stop
        while not coord.should_stop():

            # reset buffers
            action_buffer, state_buffer, reward_buffer = [], [], []

            # reset env by changing level. env.reset doesn't work for super mario
            self.env.change_level(new_level=0)
            s, _, done, _ = self.env.step(self.env.action_space.sample()) 

            # reset LSTM memory
            lstm_c = np.zeros((1, self.a3cnet.lstm_cell.state_size.c))
            lstm_h = np.zeros((1, self.a3cnet.lstm_cell.state_size.h))
            # reset LSTM memory for whole batch
            self.batch_lstm_c = lstm_c
            self.batch_lstm_h = lstm_h

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
                    self.a3cnet.s: np.expand_dims(s, axis=0),
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

                if step_counter % GLOBAL_UPDATE_INTERVAL == 0 or done:

                    """ debug """
                    print('reward for batch: ' + str(total_reward))
                    total_reward = 0
                    """ debug """

                    if done:
                        value_s = 0
                    else:
                        value_s = sess.run(
                            [self.a3cnet.critic_out],
                            feed_dict={
                                self.a3cnet.s: [s],
                                self.a3cnet.lstm_c: self.batch_lstm_c,
                                self.a3cnet.lstm_h: self.batch_lstm_h
                            }
                        )[0][0]

                    # update global net
                    print('value: ' + str(value_s))
                    print('buffer size: ' + str(len(state_buffer)))

                    # calculate discounted rewards all the way to current state s
                    # for each state in state buffer
                    discounted_rewards_buffer = []
                    discounted_reward = value_s
                    for reward in reversed(reward_buffer):
                        discounted_reward = reward + GAMMA * discounted_reward
                        discounted_rewards_buffer.insert(0, discounted_reward)
                    discounted_rewards_buffer = np.array(discounted_rewards_buffer).reshape(-1,1)


                    (self.batch_lstm_c, self.batch_lstm_h), _ = sess.run(
                        [
                            self.a3cnet.layer4_lstm_state, 
                            self.a3cnet.sync_global_network
                        ],
                        feed_dict={
                            self.a3cnet.s: np.stack(state_buffer),
                            self.a3cnet.reward: discounted_rewards_buffer,
                            self.a3cnet.action_taken: action_buffer,
                            self.a3cnet.lstm_c: self.batch_lstm_c,
                            self.a3cnet.lstm_h: self.batch_lstm_h
                        }
                    )

                    # reset buffers
                    state_buffer, action_buffer, reward_buffer = [], [], []

                    # copy global net to agent
                    sess.run([self.a3cnet.copy_global_network])



                s = s_
                step_counter += 1



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
    globalz = A3CNetwork((224,256,3), 6, 'global')
    agent = Agent('meta-SuperMarioBros-v0', 'agent_0', episode_count)

    with tf.Session() as sess:
        sess.run(tf.global_variables_initializer())
        coord = tf.train.Coordinator()

        agent.train(sess, coord)
