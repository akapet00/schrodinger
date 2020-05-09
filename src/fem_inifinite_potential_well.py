import argparse
import numpy as np
from scipy.constants import h, hbar, m_e, pi
import matplotlib.pyplot as plt
from matplotlib.legend_handler import HandlerTuple
import seaborn as sns
sns.set()

from fdm_inifinite_potential_well import psi_close_form, pdf_close_form
from utils.metrics import rmse
from utils.plotting import latexconfig

parser = argparse.ArgumentParser(description='Takes the number of finite elements.')
parser.add_argument('-n', '--num_elements',
                    type=int,
                    metavar='',
                    help='Finite Element analysis of the wave equation for\
                        an electron in the infinite quantum well based on the given\
                        number of finite elements.',
                    default=10)

def assemble_l(lhss, lhs_glob):
    if lhs_glob.shape[0]!=lhs_glob.shape[1]:
        raise Exception("Global matrix is not square matrix!") 
    else:
        for i, lhs in enumerate(lhss):
            lhs_glob[i:i+lhs.shape[0], i:i+lhs.shape[1]] += lhs
    return lhs_glob    

def assemble_r(rhss, rhs_glob):
    for i, rhs in enumerate(rhss):
        rhs_glob[i:i+rhs.shape[0], ] += rhs
    return rhs_glob

def fem(N, xmin, L, bcs):
    raise NotImplementedError('Yet to appear.')

def main():
    args = parser.parse_args()
    N = args.num_elements           # number of finite elements
    xmin = .0                       # left boundary
    L = 1.0                         # right boundary
    psi_0 = .0                      # left Dirichlet bc
    psi_L = .0                      # right Dirichlet bc
    x = np.linspace(xmin, L, N)
    _x = np.linspace(xmin, L, 1000)

    principal_quantum_numbers = [1, 2, 3]       
    nonreletivistic_energies = []
    for n in principal_quantum_numbers:
        E = n**2 * h**2 / (8 * m_e * L)
        nonreletivistic_energies.append(E)

    E, psi, pdf = fem(N, xmin, L, (psi_0, psi_L))

    fig, axs = plt.subplots(len(principal_quantum_numbers), 1, sharex='all', figsize=(7, 9))
    for i, (n, E) in enumerate(zip(principal_quantum_numbers, nonreletivistic_energies)):  
        psi_analytic = psi_close_form(x, n, L)
        pdf_analytic = pdf_close_form(x, n, L)
    
        l1, = axs[i].plot(_x, psi_close_form(_x, n, L), 'b-')
        l2, = axs[i].plot(_x, pdf_close_form(_x, n, L), 'r-')
        l3, = axs[i].plot(x, psi[:, i], 'bo')
        l4, = axs[i].plot(x, pdf[:, i], 'ro')

        axs[i].legend([l1, l2, (l3, l4)], [r'$\psi(x)$', r'$|\psi(x)|^2$', 'FDM'], 
                      handler_map={tuple: HandlerTuple(ndivide=None)})
        axs[i].set_title(f'RMSE = {np.round(rmse(pdf_analytic, pdf[:, i]), 5)}')
        axs[i].legend([l1, l2, (l3, l4)], [r'$\psi(x)$', r'$|\psi(x)|^2$', 'FDM'], 
                      handler_map={tuple: HandlerTuple(ndivide=None)})
    plt.xlabel(r'$x$')
    plt.tight_layout()
    plt.show() 
    
    plt.xlabel(r'$x$')
    plt.tight_layout()
    plt.show()  

if __name__ == "__main__":
    latexconfig()
    main()