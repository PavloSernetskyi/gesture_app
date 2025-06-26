import numpy as np

data = np.load("../data/0.npy")
print("Shape:", data.shape)         # e.g., (100, 63)
print("Sample 0:", data[0])         # First gesture's feature vector
