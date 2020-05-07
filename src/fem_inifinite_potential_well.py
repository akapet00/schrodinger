import argparse
import numpy as np
from scipy.constants import h, hbar, m_e, pi
import matplotlib.pyplot as plt
from matplotlib.legend_handler import HandlerTuple
import seaborn as sns
sns.set()

parser = argparse.ArgumentParser(description='Takes the number of finite elements.')
parser.add_argument('-n', '--num_elements',
                    type=int,
                    metavar='',
                    help='Finite Element analysis of the wave equation for\
                        an electron in the infinite quantum well based on the given\
                        number of finite elements.',
                    default=10)
                             
def rmse(true, predict):
    return np.sqrt(np.mean((true - predict)**2))

def psi_close_form(x, n, L):
    """analyitcal solution for 1-D wave eqn"""
    amp = np.sqrt(2/L)
    return amp * np.sin(n * pi / L * x)

def pdf_close_form(x, n, L):
    r"""probability of locating an electron in x \in (0, L)"""
    return psi_close_form(x, n, L)**2


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

def fem(N, xmin, L, bcs, args):
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

    sol = []
    fig, axs = plt.subplots(len(principal_quantum_numbers), 1, sharex='all', figsize=(8, 12))
    for i, (n, E) in enumerate(zip(principal_quantum_numbers, nonreletivistic_energies)):
        # hbar**2/(2*m_e) * psi_xx + E * psi = 0 
        A = hbar**2/(2*m_e)
        B = 0 
        C = E
        psi = fem(N, xmin, L, (psi_0, psi_L), (A, B, C)).flatten()
        sol.append(psi)
    
        l1, = axs[i].plot(_x, psi_close_form(_x, n, L), 'b-')
        l2, = axs[i].plot(_x, pdf_close_form(_x, n, L), 'r--')
        l3, = axs[i].plot(x, psi, 'bo', markersize=7)
        l4, = axs[i].plot(x, psi**2, 'r^', markersize=7)

        axs[i].legend([l1, l2, (l3, l4)], [r'$\psi(x)$', r'$|\psi(x)|^2$', 'FDM'], 
                      handler_map={tuple: HandlerTuple(ndivide=None)})
    
    plt.xlabel(r'$x$')
    plt.tight_layout()
    plt.show()  

if __name__ == "__main__":
    main()