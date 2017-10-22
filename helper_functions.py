import numpy as np

def preprocess_state(state):
    """ Normalize the input values to be between 0 and 1 """
    state = state[32:-16]
    return np.multiply(state, 1.0/255)

def discrete_to_multi_action(action_discrete):
    """ map dicrete action to multidiscrete vector of NES buttons
    
    This is used to be able to use a simple softmax output of the policy network
    whle being able to perform multiple actions at once, 
    for example jump and move right at the same time
    """

    mapping = {
        0: [0, 0, 0, 0, 0, 0],  # NOOP
        1: [1, 0, 0, 0, 0, 0],  # Up
        2: [0, 0, 1, 0, 0, 0],  # Down
        3: [0, 1, 0, 0, 0, 0],  # Left
        4: [0, 1, 0, 0, 1, 0],  # Left + A
        5: [0, 1, 0, 0, 0, 1],  # Left + B
        6: [0, 1, 0, 0, 1, 1],  # Left + A + B
        7: [0, 0, 0, 1, 0, 0],  # Right
        8: [0, 0, 0, 1, 1, 0],  # Right + A
        9: [0, 0, 0, 1, 0, 1],  # Right + B
        10: [0, 0, 0, 1, 1, 1],  # Right + A + B
        11: [0, 0, 0, 0, 1, 0],  # A
        12: [0, 0, 0, 0, 0, 1],  # B
        13: [0, 0, 0, 0, 1, 1],  # A + B
    }

    return mapping[action_discrete]
