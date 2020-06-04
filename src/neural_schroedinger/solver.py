import autograd
import autograd.numpy as np 
from autograd import grad, jacobian
from autograd import elementwise_grad as egrad 
from autograd.misc.flatten import flatten

from scipy.integrate import simps 
from scipy.optimize import minimize
from scipy.constants import hbar, m_e

from neural_schroedinger.timer import Timer

from neural_schroedinger.activations import (tanh, sigmoid, relu, 
                                             softplus, elu, prelu)
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
    params = [] 
    for W, B in zip(weights, biases):
        params.append(W)
        params.append(B)
    return params

def _predict(params, x, bcs, activation):
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
    for i, param in enumerate(params):
        if i%2==0:
            weights.append(param)
        else:
            biases.append(param)
        
    out = np.array(list(x))
    for W, B in zip(weights[:-1], biases[:-1]):
        out = activation(out @ W + B)
    out = out @ weights[-1] + biases[-1]
    return sum(bcs) + x*(1-x) * out

# 1st derivative of NN output
_predict_x = egrad(_predict, argrnum=1)

# 2nd derivative of NN output
_predict_xx = egrad(egrad(_predict, argnum=1), argnum=1)

class NN(object):
    """Feed-forward neural network model with integrated physical knowledge.
    This impementation is based on the paper by Lagaris, I.E et al:
    'Artificial Neural Network Methods in Quantum Mechanics'
    ArXiV link: https://arxiv.org/abs/quant-ph/9705029

    The code is heavily inspired by https://github.com/JiaweiZhuang/AM205_final

    Args
    ----
        f (callable): Right-hand-side function. 
        x (autograd.numpy.ndarray): Input spatial coordinates. 
        bcs (tuple): Tuple of boundary conditions.
        sizes (list): Number of units per layer. 
    """

    def __init__(self, f, x, bcs, sizes, activation='tanh'):
        assert bcs is not tuple, \
            'Boundary conditions must be inside the tuple.'
        assert sizes is not list, \
            'Neural network layer sizes must be inside the list.'
        assert x.shape == (x.size, 1),\
            'x must be a single column autograd.numpy.ndarray.'
        assert sizes[0] == x.shape[1], \
            'Neural network input shape is ill-defined.'        
        self.f = f 
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
        return(f'Neural ODE Solver \n'
            f'----------------- \n'
            f'Boundary condtions:        {self.bcs} \n'
            f'Neural architecture:       {self.sizes} \n'
            f'Number of training points: {self.x.size}')

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
        f = self.f 
        bcs = self.bcs
            
        y_pred = _predict(params, x, bcs, self.activation)
        y_xx_pred = _predict_xx(params, x, bcs, self.activation)

        # normalized loss function
        I = simps((y_pred**2).ravel(), x.ravel())

        _H = -hbar/(2*m_e) * y_xx_pred
        E = I * simps(np.conjugate(y_pred).ravel() * _H.ravel(), x.ravel())

        loss_normal = I * np.mean((_H.ravel() - E * y_pred.ravel())**2)
        if type(loss_normal) is autograd.numpy.numpy_boxes.ArrayBox:
            print(f'loss = {loss_normal._value}')
        else: print(f'loss = {loss_normal}')

        # # l-2 norm loss function
        # f_pred = f(x, y_pred)
        # l_2 = np.mean((y_xx_pred - f_pred)**2) 

        return loss_normal
    
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

    def fit(self, method='BFGS', maxiter=2000, tol=1e-7):
        """Train the network.
        
        Args
        ----
            method (str or callable, optional): Type of solver.
                The full list is given in the official Scipy docs
                at scipy.optimize.minimize section.
            maxiter (int, optional): Number of training iterations.
        """
        t = Timer()
        t.start()
        opt = minimize(self.loss_wrap, x0=self.flattened_params,
                        jac=jacobian(self.loss_wrap), method=method,
                        tol=tol,
                        options={'disp':True, 'maxiter':maxiter})
        t.stop()
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
            y_pred = _predict(self.params_list, x, self.bcs, self.activation)
            y_xx_pred = _predict_xx(self.params_list, x, 
                                    self.bcs, self.activation)
            return y_pred, y_xx_pred
        else:
            y_pred = predict_order2(params_list, x, y0)
            return y_pred_list