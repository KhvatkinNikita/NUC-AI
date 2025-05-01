import numpy as np

def create_sequences(data, window_size):
    """
    Create sliding window sequences
    data [np.array]: training, validation or test data
    window_size [int]: size of the window
    """
    sequences = []
    targets = []
    for i in range(len(data) - window_size):
        seq = data[i:i + window_size]
        target = data[i + window_size]
        sequences.append(seq)
        targets.append(target)
    return np.array(sequences), np.array(targets)