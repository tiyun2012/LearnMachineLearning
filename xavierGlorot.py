import numpy as np

def xavier_initialization(size):
    in_dim, out_dim = size
    limit = np.sqrt(6 / (in_dim + out_dim))
    return np.random.uniform(-limit, limit, size)

# Example usage
weights = xavier_initialization((5, 4))
