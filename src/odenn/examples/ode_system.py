import time 
import sys
sys.path.append('../')

from odenn.core import Model

import matplotlib.pyplot as plt 
import seaborn as sns; sns.set()
from scipy.integrate import solve_ivp 
import autograd.numpy as np

def f(x, y):
    '''Rossler attractor system'''
    a = 0.2
    b = 0.2
    c = 5.7
    return [y[1] - y[2], 
            y[0] + a*y[1], 
            b + y[2]*(y[0]-c)]

def main():
    # input data 
    x = np.linspace(0, 1, 40).reshape(-1, 1) 
    y0_list = [1, 5, 10]

    # solution
    sol = solve_ivp(f, [x.min(), x.max()], y0_list, 
                    t_eval=x.ravel(), method='Radau', rtol=1e-5)
    
    # ann training
    nn = Model(f, x, y0_list, n_hidden=50)
    print(nn)
    nn.reset_weights()
    start = time.time() 
    nn.train(maxiter=1000) 
    end = time.time() 
    print(f'\nTraining time: {round((end - start), 4)} s')
    nn_sol, _ = nn.predict()
    
    plt.plot(sol.t, sol.y[0], 'b', label=r'$y_0$')
    plt.plot(sol.t, sol.y[1], 'r', label=r'$y_1$')
    plt.plot(sol.t, sol.y[2], 'g', label=r'$y_2$')
    plt.plot(x, nn_sol[0], 'bo', label=r'$y_0$ prediction')
    plt.plot(x, nn_sol[1], 'ro', label=r'$y_1$ prediction')
    plt.plot(x, nn_sol[2], 'go', label=r'$y_2$ prediction')
    
    plt.xlabel(r'$x$')
    plt.ylabel(r'$y$')
    plt.legend()
    plt.show()

if __name__ == "__main__":
    main()