from agent import Agent
from threading import Thread
from A3C_network import A3CNetwork
import tensorflow as tf

# where to periodically save the model
SUMMARY_FOLDER = 'pong-pixels'
MODEL_PATH = './models/' + SUMMARY_FOLDER

def main():
    """ the main function """
    LEVEL_NAME = 'Pong-v0'
    NUMBER_OF_AGENTS = 4
    global_shape = (210, 160, 3)

    tf.reset_default_graph()


    sess = tf.Session()
    coord = tf.train.Coordinator()

    episode_count = tf.Variable(
        0, 
        dtype=tf.int32,
        name='episode_count',
        trainable=False
    )

    global_writer = tf.summary.FileWriter('./logs/%s/global' % SUMMARY_FOLDER)

    globalz = A3CNetwork(global_shape, 6, 'global')

    """ create an array of agents"""
    agents = []
    for i in range(0,NUMBER_OF_AGENTS):
        agents.append(Agent(
            LEVEL_NAME, 
            global_shape, 
            'agent_' + str(i), 
            episode_count, 
            global_writer
        ))

    """ run all agents in separate threads """
    agent_threads = []


    load_model = False

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
