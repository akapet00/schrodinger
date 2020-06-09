from . import Timer
from . import (tanh, sigmoid, relu, 
               softplus, elu, prelu)
import autograd
import autograd.numpy as np 
from autograd import grad, jacobian
from autograd import elementwise_grad as egrad 
from autograd.misc.flatten import flatten
from scipy.integrate import simps 
from scipy.optimize import minimize
from scipy.constants import hbar, m_e
import matplotlib.pyplot as plt

activation_dispatcher = {
    'tanh': tanh,
    'sigmoid': sigmoid,
    'relu': relu,
    'softplus': softplus,
    'elu': elu,
    'prelu': prelu,
}

def _init_weights(sizes):
    """Initialize weights and biases of a feed-forward 
    shallow neural network (NN). 

    Args
    ----
        sizes (list): Number of units per layer. 

    Returns
    -------
        list: weights and biases
    """

    weights = [np.random.randn(x, y) for x, y in zip(sizes[:-1], sizes[1:])]
    biases = [np.zeros(y) for y in sizes[1:]]
    b = 10.

    params = [] 
    for W, B in zip(weights, biases):
        params.append(W)
        params.append(B)
    params.append(b)
    return params

def _predict(params, x, activation):
    """Calculate the output of the NN.
    
    Args
    ----
        params (list): List of weights and biases.
        x (autograd.numpy.ndarray): Input spatial coordinates. 
        bcs (tuple): Tuple of boundary conditions.
        activation (callable, optional): Activation function.

    Returns
    -------
        float: the output of a forward pass
    """
    weights, biases = [], []
    for i, param in enumerate(params[:-1]):
        if i%2==0:
            weights.append(param)
        else:
            biases.append(param)
        
    out = np.array(list(x))
    for W, B in zip(weights[:-1], biases[:-1]):
        out = activation(out @ W + B)
    out = out @ weights[-1] + biases[-1]
    
    b = params[-1]
    return np.exp(-b * x**2) * out

# 1st derivative of NN output
_predict_x = egrad(_predict, argnum=1)

# 2nd derivative of NN output
_predict_xx = egrad(_predict_x, argnum=1)

class NN(object):
    """Feed-forward neural network model with integrated physical knowledge.
    This impementation is based on the paper by Lagaris, I.E. et al:
    'Artificial Neural Network Methods in Quantum Mechanics'
    ArXiV link: https://arxiv.org/abs/quant-ph/9705029

    The code is inspired by https://github.com/JiaweiZhuang/AM205_final

    Args
    ----
        x (autograd.numpy.ndarray): Input spatial coordinates. 
        bcs (tuple): Tuple of boundary conditions.
        sizes (list): Number of units per layer. 
    """

    def __init__(self, x, bcs, sizes, activation='tanh'):
        assert bcs is not tuple, \
            'Boundary conditions must be inside the tuple.'
        assert sizes is not list, \
            'Neural network layer sizes must be inside the list.'
        assert x.shape == (x.size, 1),\
            'x must be a single column autograd.numpy.ndarray.'
        assert sizes[0] == x.shape[1], \
            'Neural network input shape is ill-defined.'        
        self.x = x 
        self.bcs = bcs 
        self.sizes = sizes 
        try:
            self.activation = activation_dispatcher[activation]
        except KeyError:
            raise ValueError('Invalid activation function input.')
    
        self.loss = 0 
        self.reset_weights()

    def __str__(self):
        return(f'Neural Schroedinger Solver \n'
            f'------------------------------------- \n'
            f'Boundary condtions:        {self.bcs} \n'
            f'Neural architecture:       {self.sizes} \n'
            f'Number of training points: {self.x.size} \n'
            f'------------------------------------- \n')

    def __repr__(self):
        return self.__str__()

    def reset_weights(self):
        """Reset parameters of a NN."""

        self.params_list = _init_weights(sizes=self.sizes)
        self.flattened_params, self.unflat_func = flatten(self.params_list)

    def loss_fun(self, params):
        """Returns total loss for given parameters.

        Args
        ----
            params (autograd.numpy.ndarray): Unflattend weights and biases.

        Returns
        -------
            float: value of the loss function
        """

        x = self.x 
        bcs = self.bcs
            
        y_pred = _predict(params, x, self.activation)
        y_xx_pred = _predict_xx(params, x, self.activation)

        I = simps((y_pred**2).ravel(), x.ravel())
        H = - hbar**2/(2*m_e) * y_xx_pred
        E = simps((np.conj(y_pred) * H).ravel(), x.ravel()) / I

        return np.sum((H - E * y_pred)**2) / I \
            + (y_pred[0] - bcs[0])**2 + (y_pred[-1] - bcs[-1])**2
    
    def loss_wrap(self, flattened_params):
        """Unflatten the parameter list.
        
        Args
        ----
            flattened_params (autograd.numpy.ndarray): Flattend weights and 
                                                       biases.

        Returns
        -------
            autograd.numpy.numpy_boxes.ArrayBox: of the value the loss function
                                                 returns 
        """

        params_list = self.unflat_func(flattened_params) 
        return self.loss_fun(params_list) 

    def fit(self, method='BFGS', maxiter=2000, tol=1e-7, iprint=10):
        """Train the network.
        
        Args
        ----
            method (str or callable, optional): Type of solver.
                The full list is given in the official Scipy docs
                at scipy.optimize.minimize section.
            maxiter (int, optional): Number of training iterations.
            tol (float, optional): Optimizer tolerance.
            iprint (int, optional): Every iprint iteration print loss.
        """
        self.p = [] 
        global counter 
        counter = 0 
        global loss_arr
        loss_arr = []
        def print_loss(p):
            """Optimizer callback."""
            global counter
            if counter % iprint == 0:
                print(f'Iteration: {counter}\t Loss: {self.loss_wrap(p)}')
            counter += 1
            self.p.append(self.unflat_func(p))
            loss_arr.append(self.loss_wrap(p))

        t = Timer()
        t.start()
        opt = minimize(
            fun=self.loss_wrap, 
            x0=self.flattened_params,
            jac=grad(self.loss_wrap), 
            method=method,
            tol=tol,
            callback=print_loss,
            options={'disp':True, 'maxiter':maxiter})
        t.stop()
        self.loss = loss_arr
        self.flattened_params = opt.x 
        self.params_list = self.unflat_func(opt.x)
            
    def predict(self, x=None, params_list=None):
        """Generate the output with trained NN.

        Args
        ----
            x (autograd.numpy.ndarray, optional): Input spatial coordinates. 
            params (autograd.numpy.ndarray, optional): Unflattend weights 
                and biases.

        Returns
        -------
            autograd.numpy.ndarray: predictions generated over x space
        """

        if x is None:
            x = self.x 
        
        if params_list is None:
            y_pred = _predict(self.params_list, x, self.activation)
            y_xx_pred = _predict_xx(self.params_list, x, self.activation)
            return y_pred, y_xx_pred
        else:
            y_pred = _predict(params_list, x, self.activation)
            return y_pred
        
    def plot_loss(self, log=True):
        """Plot value of loss function.
        
        Args
        ----
        log (Bool, optional): y-axis in log-scale if True.
        """
        _, ax = plt.subplots()
        ax.plot(range(len(loss_arr)), loss_arr, 'k-')
        if log:
            ax.set_yscale('log')
            y_lbl = 'Log-'
        y_lbl = ''
        ax.set_xlabel('Iterations')
        ax.set_ylabel(f'{y_lbl}Loss')
        plt.grid()
        plt.show()