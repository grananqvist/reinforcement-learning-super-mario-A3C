"""
This script can be run to manually play the game.
Alternatively, remove the human wrapper and uncomment line 36
to perform a random walk
"""
from multiprocessing import Lock
import gym


__all__ = ['SetPlayingMode']

def SetPlayingMode(target_mode):
    """ target mode can be 'algo' or 'human' """

    class SetPlayingModeWrapper(gym.Wrapper):
        """
        Doom wrapper to change playing mode 'human' or 'algo'
        """
        def __init__(self, env):
            super(SetPlayingModeWrapper, self).__init__(env)
            if target_mode not in ['algo', 'human']:
                raise gym.error.Error('Error - The mode "{}" is not supported. Supported options are "algo" or "human"'.format(target_mode))
            self.unwrapped.mode = target_mode

    return SetPlayingModeWrapper


env = gym.make('meta-SuperMarioBros-v0')
wrapper = SetPlayingMode('human')
env = wrapper(env)
#env.configure(lock=Lock())
observation = env.reset()
env.render()

action = env.action_space.sample() 

for _ in range(100000000):
        env.render()

        #action = env.action_space.sample() # your agent here (this takes random actions)
        observation, reward, done, info = env.step(action)

        print(reward)
        if done:
            print('done and reward: ' + str(reward))
            print(info)

        if _ % 1000:
            print(info)
