from multiprocessing import Lock
import gym

env = gym.make('meta-SuperMarioBros-v0')
#env.configure(lock=Lock())
observation = env.reset()

env.render()

action = env.action_space.sample() # your agent here (this takes random actions)
action = [0, 0, 0, 1, 0, 0]
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

for _ in range(100000):
        env.render()

        action = env.action_space.sample() # your agent here (this takes random actions)
        observation, reward, done, info = env.step(action)

        print(reward)
        if done:
            print('done and reward: ' + str(reward))
            print(info)

        if _ % 1000:
            print(info)
