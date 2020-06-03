import autograd.numpy as np 

def sigmoid(z):
    """Sigmoid activation function implementation."""
    return 1.0/(1.0 + np.exp(-z))

def tanh(z):
    """Hyperbolic tan activation function implementation."""
    return np.tanh(z)

def relu(z, alpha=0., max_value=None, threshold=0.0):
    """Rectified linear unit activation function implementation.
    Acquired from keras.backend library and applied to autograd
    backend.
    """
    if max_value is None:
        max_value = np.inf
        above_threshold = z * (z >= threshold)
        above_threshold = np.clip(above_threshold, 0.0, max_value)
        below_threshold = alpha * (z - threshold) * (z < threshold)
    return below_threshold + above_threshold

def softplus(z):
    """Smooth approximation to the rectifier."""
    return np.log(np.ones_like(z) + np.exp(z))

def elu(z, alpha=1.0):
    """Exponential linear unit activation function implementation.
    Acquired from keras.backend.numpy_backend lib and applied to 
    autograd backend.
    """
    return z * (z > 0) + alpha * (np.exp(z) - 1.0) * (z < 0)