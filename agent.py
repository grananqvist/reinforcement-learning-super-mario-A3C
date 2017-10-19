import numpy as np
import tensorflow as tf
import gym
from multiprocessing import Lock
from A3C_network import A3CNetwork
from helper_functions import discrete_to_multi_action, preprocess_state
from PIL import Image
from random import randint

# max distance for each level 0-31
MAX_DISTANCE_LEVEL = [
    3266, 3298, 3298, 3698, 3282, 3106, 2962,   
    6114, 3266, 3266, 3442, 3266, 3298, 3554,
    3266, 3554, 2514, 3682, 2498, 2434, 2514, 2754,
    3682, 3554, 2430, 2430, 2430, 2942, 2429, 2429, 3453, 4989]

# discount factor
GAMMA = 0.99

# number of iterations to store in buffers before updating global net
GLOBAL_UPDATE_INTERVAL = 30

# where to periodically save the model
SUMMARY_FOLDER = 'mario-pixel-models-restart'
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

        # Unlock all levels
        #for i,l in enumerate(self.env.locked_levels):
            #self.env.locked_levels[i] = False
        current_level = 0

        restart = False

        # episode loop. Continue playing the game while should not stop
        while not coord.should_stop():

            # reset buffers
            action_buffer, state_buffer, reward_buffer, value_buffer = [], [], [], []

            # reset env by changing level. env.reset doesn't work for super mario
            # also change to the latest unlocked level
            s, _, done, info = self.env.step(self.env.action_space.sample()) 

            """if info['level'] < current_level:
                self.env.locked_levels[current_level] = False   # Unlock level
                self.env.change_level(new_level=current_level)  # Change level
            """
            latest_level = np.argmax(info['locked_levels']) - 1
            if latest_level > 0:
                self.env.unwrapped.change_level(new_level=0)
            #s = preprocess_state(s)
            s = np.expand_dims(s, axis=2)

            prev_score = 0
            prev_time = 400

            # reset LSTM memory
            lstm_c = np.zeros((1, self.a3cnet.lstm_cell.state_size.c))
            lstm_h = np.zeros((1, self.a3cnet.lstm_cell.state_size.h))
            # reset LSTM memory for whole batch
            self.batch_lstm_c = lstm_c
            self.batch_lstm_h = lstm_h

            # keep track of total reward for entire episode
            total_reward = 0 #TODO only for debug
            episode_reward = 0
            max_distance = 0
            steps_since_progress = 0
            restart = False

            # state step loop
            while not (done or restart):
                
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
                #s_ = preprocess_state(s_)
                s_ = np.expand_dims(s_, axis=2)

                is_stuck = False

                # save variables for summary later
                if 'life' in info:
                    current_life= info['life']
                if 'score' in info:
                    current_score = info['score']

                """ reward modifications """

                r /= 2

                # -1 reward for mario dying
                if done and 'life' in info and info['life'] == 0:
                    r -= 1
                    print('died')

                # maximum reward for gaining any score is currently clipped at 1
                if 'score' in info:
                    r += np.min([0.5, 0.01 * (info['score'] - prev_score)])
                    prev_score = info['score']

                if 'distance' in info: 
                    if max_distance < info['distance']:
                        max_distance = info['distance']
                        steps_since_progress = 0
                    else:
                        steps_since_progress += 1
                    
                    is_stuck = True if steps_since_progress > 50 else False


                if 'time' in info:
                    # 3 times more costly per time step if mario is stuck
                    #time_multiplier = 0.3 if is_stuck else 0.1
                    r -= 0.01 * (prev_time - info['time'])

                    prev_time = info['time']

                if steps_since_progress > 300:
                    done = True
                    current_life = 0
                    r -= 1
                    self.env.unwrapped.change_level(new_level=latest_level)
                

                # observe results and store in buffers
                state_buffer.append(s)
                action_buffer.append(action_discrete_onehot)
                reward_buffer.append(r)
                value_buffer.append(value[0,0])
                
                total_reward += r
                episode_reward += r

                # Check if level should be changed
                if done and 'distance' in info and info['distance'] > 0.95*MAX_DISTANCE_LEVEL[info['level']]:
                    current_level += 1

                if step_counter % GLOBAL_UPDATE_INTERVAL == 0 or done or restart:

                    """ debug """
                    #print(str(self.name) + ' reward for batch: ' + str(total_reward))
                    total_reward = 0

                    #print('recent value: %f' % value)
                    #print('recent policy: ' + str(policy))
                    """ debug """

                    if done:
                        value_s = 0
                    else:
                        value_s = sess.run(
                            [self.a3cnet.critic_out],
                            feed_dict={
                                self.a3cnet.s: [s_],
                                self.a3cnet.lstm_c: lstm_c,
                                self.a3cnet.lstm_h: lstm_h
                            }
                        )[0][0]

                    # update global net
                    #print('buffer size: ' + str(len(state_buffer)))

                    # calculate discounted rewards all the way to current state s
                    # for each state in state buffer
                    discounted_rewards_buffer = []
                    discounted_reward = value_s
                    for reward in reversed(reward_buffer):
                        discounted_reward = reward + GAMMA * discounted_reward
                        discounted_rewards_buffer.insert(0, discounted_reward)
                    discounted_rewards_buffer = np.array(discounted_rewards_buffer).reshape(-1,1)
                    
                    # calculate advantages
                    discounted_advantages_buffer = []
                    advantages_buffer = np.array(reward_buffer) + \
                            GAMMA * np.array( value_buffer[1:] + [value_s] ) - \
                            np.array(value_buffer)

                    discounted_advantage = 0
                    for advantage in reversed(advantages_buffer):
                        discounted_advantage = advantage + GAMMA * discounted_advantage
                        discounted_advantages_buffer.insert(0, discounted_advantage)
                    discounted_advantages_buffer = np.array(discounted_advantages_buffer).reshape(-1,1)



                    (self.batch_lstm_c, self.batch_lstm_h), summary, _ = sess.run(
                        [
                            self.a3cnet.layer4_lstm_state, 
                            self.a3cnet.summary_op,
                            self.a3cnet.sync_global_network
                        ],
                        feed_dict={
                            self.a3cnet.s: np.stack(state_buffer),
                            self.a3cnet.reward: discounted_rewards_buffer,
                            self.a3cnet.advantage: discounted_advantages_buffer,
                            self.a3cnet.action_taken: action_buffer,
                            self.a3cnet.lstm_c: self.batch_lstm_c,
                            self.a3cnet.lstm_h: self.batch_lstm_h
                        }
                    )

                    # copy global net to agent
                    sess.run([self.a3cnet.copy_global_network])

                    # reset buffers
                    state_buffer, action_buffer, reward_buffer, value_buffer = [], [], [], []


                    self.writer.add_summary(summary, step_counter)

                s = s_
                step_counter += 1



            # increase the episode counter
            sess.run([self.episode_count_inc])

            # print current global episode count
            global_ep = sess.run([self.episode_count])[0]
            print('%s: episode nr: %i completed' % (self.name, global_ep))

            # add distribution of weights to summary for this episode
            #summary_hist = sess.run([self.a3cnet.weights_summary_op], feed_dict={})[0]
            #self.global_writer.add_summary(summary_hist, global_ep)

            summary = tf.Summary()

            # track the total reward recieved for the finished episode
            summary.value.add(tag='Reward', simple_value=float(episode_reward))

            # track whether or not the level was completed
            is_level_completed = current_life > 0
            summary.value.add(tag='Level_completed', simple_value=float(is_level_completed))

            # track the distance mario reached
            summary.value.add(tag='Distance', simple_value=float(max_distance))

            # track the number of score gathered
            summary.value.add(tag='Score', simple_value=float(current_score))

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
