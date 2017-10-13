from agent import Agent
from threading import Thread
from A3C_network import A3CNetwork
import tensorflow as tf

# where to periodically save the model
MODEL_PATH = './mario-pixel-models'

def main():
    """ the main function """
    LEVEL_NAME = 'meta-SuperMarioBros-v0'
    NUMBER_OF_AGENTS = 4
    EPISODES = 5

    tf.reset_default_graph()

    globalz = A3CNetwork((224,256,3), 14, 'global')

    sess = tf.Session()
    coord = tf.train.Coordinator()

    episode_count = tf.Variable(
        0, 
        dtype=tf.int32,
        name='episode_count',
        trainable=False
    )

    """ create an array of agents"""
    agents = []
    for i in range(0,NUMBER_OF_AGENTS):
        agents.append(Agent(LEVEL_NAME, 'agent_' + str(i), episode_count))

    """ run all agents in separate threads """
    agent_threads = []


    load_model = True

    # saver for saving model
    saver = tf.train.Saver(max_to_keep=3)

    if load_model == True:
        print ('Loading Model...')
        ckpt = tf.train.get_checkpoint_state(MODEL_PATH)
        saver.restore(sess,ckpt.model_checkpoint_path)
    else:
        sess.run(tf.global_variables_initializer())


    for agent in agents:
        f = lambda: agent.train(sess, coord, saver)
        t = Thread(target=f)
        t.start()
        agent_threads.append(t)



if __name__ == '__main__':
    main()
