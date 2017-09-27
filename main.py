from multiprocessing import Lock
import gym

env = gym.make('SuperMarioBros-1-1-v0')
env.configure(lock=Lock())
observation = env.reset()

env.render()

action = env.action_space.sample() # your agent here (this takes random actions)
observation, reward, done, info = env.step(action)

print(env.action_space)
print(env.observation_space)
print(env.observation_space.high)
print(env.observation_space.low)
print('from step')

print(action)
print(observation)
print(reward)
print(done)
print(info)

for _ in range(10000):
        env.render()

        action = env.action_space.sample() # your agent here (this takes random actions)
        observation, reward, done, info = env.step(action)
