from agent import Agent
from threading import Thread
from A3C_network import A3CNetwork
import tensorflow as tf

# where to periodically save the model
SUMMARY_FOLDER = 'mario-pixel-models-time-pen'
MODEL_PATH = './models/' + SUMMARY_FOLDER

def main():
    """ the main function """

    LEVEL_NAME = 'meta-SuperMarioBros-v0'
    NUMBER_OF_AGENTS = 4

    # state space shape
    global_shape = (176,256,3)

    # reset graph, create tf session
    tf.reset_default_graph()
    sess = tf.Session()
    coord = tf.train.Coordinator()

    # global counter that keeps track of total number of episodes
    episode_count = tf.Variable(
        0, 
        dtype=tf.int32,
        name='episode_count',
        trainable=False
    )

    # writer for writing global summaries
    global_writer = tf.summary.FileWriter('./logs/%s/global' % SUMMARY_FOLDER)

    # the global master network that all agents send gradients to
    globalz = A3CNetwork(global_shape, 14, 'global')

    # create an array of initialized agents
    agents = []
    for i in range(0,NUMBER_OF_AGENTS):
        agents.append(Agent(
            LEVEL_NAME, 
            global_shape, 
            'agent_' + str(i), 
            episode_count, 
            global_writer
        ))

    agent_threads = []

    # whether or not to load an existing model
    # will give an error if it tries to load a model that doesn't exist
    load_model = True

    # saver for saving model
    saver = tf.train.Saver(max_to_keep=3)

    if load_model == True:
        print ('Loading Model...')
        ckpt = tf.train.get_checkpoint_state(MODEL_PATH)
        saver.restore(sess,ckpt.model_checkpoint_path)
    else:
        sess.run(tf.global_variables_initializer())

    # start the training loops of all initialized agents
    for agent in agents:
        f = lambda: agent.train(sess, coord, saver)
        t = Thread(target=f)
        t.start()
        agent_threads.append(t)


if __name__ == '__main__':
    main()
