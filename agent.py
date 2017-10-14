import numpy as np
import tensorflow as tf
import gym
from multiprocessing import Lock
from A3C_network import A3CNetwork
from helper_functions import discrete_to_multi_action, preprocess_state
from PIL import Image

# discount factor
GAMMA = 0.99

# number of iterations to store in buffers before updating global net
GLOBAL_UPDATE_INTERVAL = 30

# where to periodically save the model
SUMMARY_FOLDER = 'mario-pixel-models'
MODEL_PATH = './models/' + SUMMARY_FOLDER

class Agent(object):

    def __init__(self, level_name, global_shape, agent_name, episode_count, global_writer):

        # file writer for tensorboard
        self.writer = tf.summary.FileWriter('./logs/%s/%s' % (SUMMARY_FOLDER, agent_name))

        # global file writer to write episode metrics
        self.global_writer = global_writer

        # operation for increasing the global episode count
        self.episode_count_inc = tf.assign(episode_count, episode_count + 1)

        # unique agent name
        self.name = agent_name

        # number of total global episodes 
        self.episode_count = episode_count

        # create super mario environment
        self.env = gym.make(level_name)
        #self.env.configure(lock=Lock())
        self.state_n = global_shape #self.env.observation_space.shape
        self.action_n = 14 #self.env.action_space.shape

        # initiate A3C network
        self.a3cnet = A3CNetwork(self.state_n, self.action_n, agent_name)


    def train(self, sess, coord, saver):
        """ performs the main training loop """

        print(self.name + ' is running')

        step_counter = 1

        s = self.env.reset()

        # episode loop. Continue playing the game while should not stop
        while not coord.should_stop():

            # reset buffers
            action_buffer, state_buffer, reward_buffer = [], [], []

            # reset env by changing level. env.reset doesn't work for super mario
            self.env.change_level(new_level=2)
            s, _, done, _ = self.env.step(self.env.action_space.sample()) 
            s = preprocess_state(s)
            prev_score = 0

            # reset LSTM memory
            lstm_c = np.zeros((1, self.a3cnet.lstm_cell.state_size.c))
            lstm_h = np.zeros((1, self.a3cnet.lstm_cell.state_size.h))
            # reset LSTM memory for whole batch
            self.batch_lstm_c = lstm_c
            self.batch_lstm_h = lstm_h

            # keep track of total reward for entire episode
            total_reward = 0 #TODO only for debug
            episode_reward = 0


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
                #action = np.zeros(policy.shape[1], dtype=int)
                #action = np.array([np.random.choice(2, p=[1-policy[0,i], policy[0,i]]) 
                #    for i in range(policy.shape[1])])
                #action[np.random.choice(range(policy.shape[1]), p=policy[0])] = 1
                action_discrete = np.random.choice(range(policy.shape[1]), p=policy[0])
                action_discrete_onehot = np.zeros(self.action_n, dtype=int)
                action_discrete_onehot[action_discrete] = 1
                action = discrete_to_multi_action(action_discrete)

                # take a step in env with action
                s_, r, done, info = self.env.step(action)
                s_ = preprocess_state(s_)

                """ reward modifications """

                # -1 reward for mario dying
                if done and 'life' in info and info['life'] == 0:
                    r -= 1

                # maximum reward for gaining any score is currently clipped at 1
                r += np.min([1, 0.005 * (info['score'] - prev_score)])
                

                # observe results and store in buffers
                state_buffer.append(s)
                action_buffer.append(action_discrete_onehot)
                reward_buffer.append(r)
                
                total_reward += r
                episode_reward += r

                if step_counter % GLOBAL_UPDATE_INTERVAL == 0 or done:

                    """ debug """
                    print(str(self.name) + ' reward for batch: ' + str(total_reward))
                    total_reward = 0

                    print('recent value: %f' % value)
                    print('recent policy: ' + str(policy))
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
                    print('buffer size: ' + str(len(state_buffer)))

                    # calculate discounted rewards all the way to current state s
                    # for each state in state buffer
                    discounted_rewards_buffer = []
                    discounted_reward = value_s
                    for reward in reversed(reward_buffer):
                        discounted_reward = reward + GAMMA * discounted_reward
                        discounted_rewards_buffer.insert(0, discounted_reward)
                    discounted_rewards_buffer = np.array(discounted_rewards_buffer).reshape(-1,1)


                    (self.batch_lstm_c, self.batch_lstm_h), summary, _ = sess.run(
                        [
                            self.a3cnet.layer4_lstm_state, 
                            self.a3cnet.summary_op,
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

                    # copy global net to agent
                    sess.run([self.a3cnet.copy_global_network])

                    # reset buffers
                    state_buffer, action_buffer, reward_buffer = [], [], []


                    self.writer.add_summary(summary, step_counter)

                s = s_
                prev_score = info['score']
                step_counter += 1



            # increase the episode counter
            sess.run([self.episode_count_inc])

            # print current global episode count
            global_ep = sess.run([self.episode_count])[0]
            print('%s: episode nr: %i completed' % (self.name, global_ep))

            # add distribution of weights to summary for this episode
            summary_hist = sess.run([self.a3cnet.weights_summary_op], feed_dict={})[0]
            print('weights:')
            print(summary_hist)
            self.global_writer.add_summary(summary_hist, global_ep)

            # track the total reward recieved for the finished episode
            summary = tf.Summary()
            summary.value.add(tag='Reward', simple_value=float(episode_reward))
            self.global_writer.add_summary(summary, global_ep)
            self.global_writer.flush()

            # save model every 5 global episode
            if global_ep % 4 == 0:
                saver.save(sess, '%s/model-%i.ckpt' % (MODEL_PATH, global_ep))
                print('saved model at episode %i' % global_ep)


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
