import numpy as np

def batch_norm(X, epsilon=1e-5):
    mean = np.mean(X, axis=0, keepdims=True)
    var = np.var(X, axis=0, keepdims=True)
    X_norm = (X - mean) / np.sqrt(var + epsilon)
    return X_norm