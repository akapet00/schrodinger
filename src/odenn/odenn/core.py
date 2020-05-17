import autograd.numpy as np 
from autograd import grad 
from autograd import elementwise_grad as egrad 
from autograd.misc.flatten import flatten 
from scipy.optimize import minimize 
from scipy.integrate import solve_ivp, solve_bvp

def init_weights(n_in=1, n_hidden=10, n_out=1):
    W1 = np.random.randn(n_in, n_hidden)
    b1 = np.zeros(n_hidden) 
    W2 = np.random.randn(n_hidden, n_out) 
    b2 = np.zeros(n_out)
    params = [W1, b1, W2, b2]
    return params 

def predict(params, x, y0, activate=np.tanh):
    W1, b1, W2, b2 = params 
    out = activate(x @ W1 + b1) @ W2 + b2
    return y0 + x*out # ref: Lagaris et al (1998), 1st order

def predict_order2(params, x, y0, yL, activate=np.tanh):
    '''Supports Dirichlet conditions.'''
    W1, b1, W2, b2 = params 
    out = activate(x @ W1 + b1) @ W2 + b2
    return y0*(1-x) + yL*x + x*(1-x)*out # ref: Lagaris et al (1998), 2nd order

predict_dx = egrad(predict, argnum=1)

predict_dx_order2 = egrad(predict_order2, argnum=1)
predict_dxdx = egrad(egrad(predict_order2, argnum=1), argnum=1)

class Model(object):
    def __init__(self, f, x, y0s, yLs=[None], n_hidden=10):
        num_of_eqns = len(y0s)
        assert len(f(x[0], y0s)) == num_of_eqns, \
            'f and y0_list should have same size'
        assert x.shape == (x.size, 1), 't must be a column vector'

        self.Nvar = num_of_eqns
        self.f = f 
        self.x = x 
        self.y0_list = y0s
        self.yL_list = yLs
        self.n_hidden = n_hidden 
        self.loss = 0 
        self.reset_weights() 

    def __str__(self):
        return(f'Neural ODE Solver \n'
            f'----------------- \n'
            f'Number of equations:       {self.Nvar} \n'
            f'Initial conditions:        {self.y0_list} \n'
            f'Right boundary condtions:  {self.yL_list} \n'
            f'Number of hidden units:    {self.n_hidden} \n'
            f'Number of training points: {self.x.size}')
    
    def __repr__(self):
        return self.__str__()

    def reset_weights(self):
        self.params_list = [init_weights(n_hidden=self.n_hidden)
                            for _ in range(self.Nvar)]

        self.flattened_params, self.unflat_func = flatten(self.params_list)
    
    def loss_func(self, params_list):
        y0_list = self.y0_list 
        yL_list = self.yL_list
        x = self.x 
        f = self.f 

        y_pred_list = [] 
        dydx_pred_list = [] 
        dydxdx_pred_list = []
        
        if yL_list[0] is None:
            for params, y0 in zip(params_list, y0_list):            
                y_pred_list.append(predict(params, x, y0)) 
                dydx_pred_list.append(predict_dx(params, x, y0)) 

            f_pred_list = f(x, y_pred_list)

            loss_total = 0.0             
            for f_pred, dydx_pred in zip(f_pred_list, dydx_pred_list):
                loss = np.mean((dydx_pred - f_pred)**2) 
                loss_total += loss 
        else:
            for params, y0, yL in zip(params_list, y0_list, yL_list):            
                y_pred_list.append(predict_order2(params, x, y0, yL)) 
                dydx_pred_list.append(predict_dx_order2(params, x, y0, yL)) 
                dydxdx_pred_list.append(predict_dxdx(params, x, y0, yL))
            
            f_pred_list = f(x, y_pred_list)

            loss_total = 0.0             
            for f_pred, dydxdx_pred in zip(f_pred_list, dydxdx_pred_list):
                loss = np.mean((dydxdx_pred - f_pred)**2) 
                loss_total += loss 
        return loss_total
    
    def loss_wrap(self, flattened_params):
        params_list = self.unflat_func(flattened_params) 
        return self.loss_func(params_list) 

    def train(self, method='BFGS', maxiter=2000):
        opt = minimize(self.loss_wrap, x0=self.flattened_params,
                        jac=grad(self.loss_wrap), method=method,
                        options={'disp':True, 'maxiter':maxiter})
        self.flattened_params = opt.x 
        self.params_list = self.unflat_func(opt.x)

    def predict(self, x=None, params_list=None):
        if x is None:
            x = self.x 
        if params_list is None:
            y_pred_list = [] 
            dydx_pred_list = [] 
            for params, y0 in zip(self.params_list, self.y0_list):
                y_pred = predict(params, x, y0)
                dydx_pred = predict_dx(params, x, y0)
                y_pred_list.append(y_pred.squeeze()) 
                dydx_pred_list.append(dydx_pred.squeeze())
            return y_pred_list, dydx_pred_list 
        else:
            y_pred_list = [] 
            for params, y0 in zip(params_list, self.y0_list):
                y_pred = predict(params, t, y0)
                y_pred_list.append(y_pred.squeeze())
            return y_pred_list
    
    def predict_order2(self, x=None, params_list=None):
        if x is None:
            x = self.x 
        if params_list is None:
            y_pred_list = [] 
            dydxdx_pred_list = []
            for params, y0, yL in zip(self.params_list, self.y0_list, self.yL_list):
                y_pred = predict_order2(params, x, y0, yL)
                dydxdx_pred = predict_dxdx(params, x, y0, yL)
                y_pred_list.append(y_pred.squeeze()) 
                dydxdx_pred_list.append(dydxdx_pred.squeeze())
            return y_pred_list, dydxdx_pred_list 
        else:
            y_pred_list = [] 
            for params, y0, yL in zip(params_list, self.y0_list, self.yL_list):
                y_pred = predict_order2(params, x, y0)
                y_pred_list.append(y_pred.squeeze())
            return y_pred_list