import time 
import sys
sys.path.append('../')

from odenn.core import Model

import matplotlib.pyplot as plt 
import seaborn as sns; sns.set()
from scipy.integrate import solve_ivp 
import autograd.numpy as np

def f(x, y):
    '''
    dy/dx = y(x)
    '''
    return [y[0]]

def analytic_sol(x):
    return np.exp(x)

def main():
    # input data 
    x = np.linspace(0, 1, 20).reshape(-1, 1)
    y0_list = [1]   # initial condition: y(x=0) = 1

    # solution
    anal_sol = analytic_sol(x)
    rk_sol = solve_ivp(f, [x.min(), x.max()], y0_list, method='RK45', rtol=1e-5)

    # ann training
    nn = Model(f, x, y0_list)
    print(nn)
    start = time.time() 
    nn.train() 
    end = time.time() 
    print(f'\nTraining time: {round((end - start), 4)} s')
    nn_sol, _ = nn.predict()
    
    plt.plot(x, anal_sol, 'b', label='Analytic')
    plt.plot(rk_sol.t, rk_sol.y[0], 'bo', label='Runge-Kutta')
    plt.plot(x, nn_sol[0], 'rx', label='NN')

    plt.xlabel(r'$x$')
    plt.ylabel(r'$y$')
    plt.title(r'$dy/dx = y(x)$')
    plt.legend()
    plt.show()

if __name__ == "__main__":
    main()