
class Agent(object):
  
  def __init__(self):
    pass
    
    """ unique instance of env """

    """ copy of global network """


  def train(self):
    """ performs the main training loop """

    pass

    """ EPISODE LOOP - while processes should not stop, perform train another episode """

    """ TRAIN ONE STEP LOOP """

    """ perform action, save state & reward to buffer """

    """ if should update global network
            calculate discounted rewards
            update global network 
            pull new copy of global network

    """

    """ if done
          calculate some metrics for tensorboard
          break the TRAIN ONE STEP LOOP to start another episode
    """


