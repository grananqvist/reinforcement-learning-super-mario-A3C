from agent import Agent
from threading import Thread
from A3C_network import A3CNetwork
import tensorflow as tf

def main():
    """ the main function """
    LEVEL_NAME = 'meta-SuperMarioBros-v0'
    NUMBER_OF_AGENTS = 4
    EPISODES = 5

    episode_count = tf.Variable(
        0, 
        dtype=tf.int32,
        name='episode_count',
        trainable=False
    )

    globalz = A3CNetwork((224,256,3), 14, 'global')

    """ create an array of agents"""
    agents = []
    for i in range(0,NUMBER_OF_AGENTS):
        agents.append(Agent(LEVEL_NAME, 'agent_' + str(i), episode_count))

    """ run all agents in separate threads """
    agent_threads = []


    sess = tf.Session()
    sess.run(tf.global_variables_initializer())
    coord = tf.train.Coordinator()
    for agent in agents:
        f = lambda: agent.train(sess, coord)
        t = Thread(target=f)
        t.start()
        agent_threads.append(t)



if __name__ == '__main__':
    main()
