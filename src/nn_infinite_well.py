import time
import sys

from odenn.core import Model

import matplotlib.pyplot as plt
import seaborn as sns; sns.set()
import autograd.numpy as np
from scipy.constants import h, hbar, m_e

quantum_nmbs = [1, 2, 3]

def main():
    # input data 
    L = 1
    x = np.linspace(0, L, 50).reshape(-1, 1)
    y0_list = [0]     # initial condition: y(x=0) = 1
    yL_list = [0]     # right boundary condition: y(x=L) = e + 1/e

    _, axs = plt.subplots(len(quantum_nmbs), 1, sharex='all')
    for n in quantum_nmbs:
        def f(x, y):
            '''
            hbar^2/m_e psi_xx = E psi
            '''
            return [-(n**2 * h**2 / (8 * m_e))*2*m_e/hbar**2*y[0]]
        # ann training
        nn = Model(f, x, y0_list, yL_list, n_hidden=10)
        print(nn)
        start = time.time() 
        nn.train(maxiter=500) 
        end = time.time() 
        print(f'\nTraining time: {round((end - start), 4)} s')
        nn_sol, _ = nn.predict_order2()
        _, = axs[n-1].plot(x, nn_sol[0]**2, label=f'$n$={n}')
        axs[n-1].legend()

    plt.xlabel(r'$x$')
    plt.show()

if __name__ == "__main__":
    main()
