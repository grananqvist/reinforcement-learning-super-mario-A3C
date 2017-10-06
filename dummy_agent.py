from multiprocessing import Lock
import gym

class Agent():

    def __init__(self, level):

        self.env = gym.make(level)
        self.env.configure(lock=Lock())
        """ Start fceux"""
        self.env.reset()


    def train(self, episodes):
        """ Performs main training loop """

        for episode_num in range(0,episodes):
            print("Start episode %s" %  (episode_num))

            done = False
            while not done:

                """ Do random actions """
                action = self.env.action_space.sample()
                observation, reward, done, info = self.env.step(action)

            """ After done=True, env resets automatically """

