import time 
import sys
sys.path.append('../')

from odenn.core import Model

import matplotlib.pyplot as plt 
import seaborn as sns; sns.set()
import autograd.numpy as np

def f(x, y):
    '''
    y_xx = y(x)
    '''
    return [y[0]]

def analytic_sol(x):
    return np.exp(x) + np.exp(-x)

def main():
    # input data 
    L = 1
    x = np.linspace(0, L, 20).reshape(-1, 1)
    y0_list = [2]                           # initial condition: y(x=0) = 1
    yL_list = [np.exp(1) + 1/np.exp(1)]     # right boundary condition: y(x=L) = e + 1/e
   
    # ann training
    nn = Model(f, x, y0_list, yL_list, n_hidden=50)
    print(nn)
    start = time.time() 
    nn.train() 
    end = time.time() 
    print(f'\nTraining time: {round((end - start), 4)} s')
    nn_sol, _ = nn.predict_order2()
    
    plt.plot(x, analytic_sol(x), 'b', label='Analytic')
    plt.plot(x, nn_sol[0], 'rx', label='NN')

    plt.xlabel(r'$x$')
    plt.ylabel(r'$y$')
    plt.title(r'$d^2y/dx^2 = y(x)$')
    plt.legend()
    plt.show()

if __name__ == "__main__":
    main()