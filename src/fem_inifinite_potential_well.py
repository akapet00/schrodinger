import argparse
import numpy as np
from scipy.constants import h, hbar, m_e, pi
from scipy.linalg import eigh
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

def assemble_quad(lhs_loc, lhs_glob):
    if lhs_glob.shape[0]!=lhs_glob.shape[1]:
        raise Exception("Global matrix is not square matrix!") 
    else:
        for i, lhs in enumerate(lhs_loc):
            lhs_glob[i:i+lhs.shape[0], i:i+lhs.shape[1]] += lhs
    return lhs_glob    

def fem(N, xmin, L, bcs):
    dx = (L - xmin)/N
    xmesh = np.linspace(0, L, N+1)

    lhs_glob = np.zeros((N+1, N+1))
    rhs_glob = np.zeros((N+1, N+1))

    # matrix elems
    a_11 = 1/dx
    a_12 = -1/dx
    a_21 = a_12 
    a_22 = a_11

    b_11 = dx/3 
    b_12 = dx/6 
    b_21 = b_12 
    b_22 = b_11

    # utilizing local lhs and rhs matrices
    lhs = np.zeros((2, 2))
    rhs = np.zeros((2, 2))
    lhs_loc = []
    rhs_loc = []
    a = 0
    b = dx
    for _ in range(N):
        lhs[0,0] = a_11
        lhs[0,1] = a_12
        lhs[1,0] = a_21
        lhs[1,1] = a_22
        lhs_loc.append(lhs)

        rhs[0,0] = b_11
        rhs[0,1] = b_12
        rhs[1,0] = b_21
        rhs[1,1] = b_22
        rhs_loc.append(rhs)

        a = a + dx
        b = b + dx

    # assembling global lhs and rhs matrix
    lhs_glob = assemble_quad(lhs_loc, lhs_glob)
    rhs_glob = assemble_quad(rhs_loc, rhs_glob)

    # Weak Galerking-Bubnov scheme without natural boundary condition
    lhs_glob = lhs_glob * (hbar**2)/(2*m_e)
 
    # applying dirichlet conds
    if bcs == (0, 0):
        lhs_glob = lhs_glob[1:-1, 1:-1]
        rhs_glob = rhs_glob[1:-1, 1:-1]
    else:
        raise ValueError

    # eigenvalues (E) and eigenvector (psi)
    E, psi = eigh(lhs_glob, rhs_glob)
    psi = np.r_[np.zeros((1, psi.shape[1])), psi]
    psi = np.r_[psi, np.zeros((1, psi.shape[1]))]

    # probability density function |psi|^2
    pdf = psi * np.conj(psi)
    return E, psi, pdf

def main():
    args = parser.parse_args()
    N = args.num_elements           # number of finite elements
    xmin = .0                       # left boundary
    L = 1.0                         # right boundary
    psi_0 = .0                      # left Dirichlet bc
    psi_L = .0                      # right Dirichlet bc
    x = np.linspace(xmin, L, N+1)
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
        if i == 0: # stupid hack, should be fixed 
            l3, = axs[i].plot(x, -psi[:, i], 'bo', markersize=5, alpha=0.8)
        else:
            l3, = axs[i].plot(x, psi[:, i], 'bo', markersize=5, alpha=0.8)
        l4, = axs[i].plot(x, pdf[:, i], 'ro', markersize=5, alpha=0.8)

        axs[i].legend([l1, l2, (l3, l4)], [r'$\psi(x)$', r'$|\psi(x)|^2$', 'FEM'], 
                      handler_map={tuple: HandlerTuple(ndivide=None)})
        axs[i].set_title(f'RMSE = {np.round(rmse(pdf_analytic, pdf[:, i]), 5)}')
        axs[i].legend([l1, l2, (l3, l4)], [r'$\psi(x)$', r'$|\psi(x)|^2$', 'FEM'], 
                      handler_map={tuple: HandlerTuple(ndivide=None)})
    plt.xlabel(r'$x$')
    plt.tight_layout()
    plt.show() 

if __name__ == "__main__":
    latexconfig()
    main()