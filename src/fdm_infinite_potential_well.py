import argparse
import numpy as np
from scipy.constants import h, hbar, m_e, pi
from scipy.linalg import eigh
import matplotlib.pyplot as plt
from matplotlib.legend_handler import HandlerTuple
import seaborn as sns
sns.set()
from utils.plotting import latexconfig
from utils.metrics import rmse

parser = argparse.ArgumentParser(description='Takes the number of grid points.')
parser.add_argument('-n', '--num_of_points',
                    type=int,
                    metavar='',
                    help='Finite Difference approximation of the wave equation for\
                        an electron in the infinite quantum well based on the given\
                        number of grid points.',
                    default=10)

def psi_close_form(x, n, L):
    """analyitcal solution for 1-D wave eqn"""
    amp = np.sqrt(2/L)
    return amp * np.sin(n * pi / L * x)

def pdf_close_form(x, n, L):
    r"""probability of locating an electron in x \in (0, L)"""
    return psi_close_form(x, n, L)**2

def V(x):
    """Potential function"""
    return 0. * x

def fdm(N, xmin, L, bcs):
    """(- hbar**2/(2*m_e) * d^2/dx^2) psi = E * psi"""
    dx = (L - xmin)/(N-1)           # distance between grid points
    xmesh = np.linspace(xmin, L, N) # mesh    

    # hamiltonian formulation
    d_xx = np.diag(np.full(N, 2), k=0) +\
          np.diag(np.full(N-1, -1), k=-1) +\
          np.diag(np.full(N-1, -1), k=+1)
    H = hbar/(2 * m_e * dx**2) * d_xx + np.diag(np.full(N, V(xmesh)), k=0)

    # applying dirichlet conds
    if bcs == (0, 0):
        H = H[1:-1, 1:-1]
        # eigenvalues (E) and eigenvector (psi)
        E, psi = eigh(H)
        psi = np.r_[np.zeros((1, psi.shape[1])), psi]
        psi = np.r_[psi, np.zeros((1, psi.shape[1]))]
    else:
        raise ValueError

    # eigenvector normalization
    psi = psi * np.linalg.norm(psi)
    
    # probability density function |psi|^2
    pdf = psi * np.conj(psi)
    return E, psi[::-1], pdf
    
def main():
    args = parser.parse_args()
    N = args.num_of_points          # number of grid points
    xmin = 0.0                      # left boundary
    L = 1.0                         # right boundary
    psi_0 = 0                       # left Dirichlet bc
    psi_L = 0                       # right Dirichlet bc
    x = np.linspace(xmin, L, N)
    _x = np.linspace(xmin, L, 1000)

    principal_quantum_numbers = list(range(1, 4))
    nonreletivistic_energies = []
    for n in principal_quantum_numbers:
        E = n**2 * h**2 / (8 * m_e * L)
        nonreletivistic_energies.append(E)

    # H psi = E psi
    # H = - hbar**2/(2*m_e) * d^2/dx^2 -> eigenvalue problem
    E, psi, pdf = fdm(N, xmin, L, bcs=(psi_0, psi_L))
    
    fig, axs = plt.subplots(len(principal_quantum_numbers), 1, sharex='all', figsize=(7, 9))
    for i, (n, E_analytic) in enumerate(zip(principal_quantum_numbers, nonreletivistic_energies)):    
        l1, = axs[i].plot(_x, psi_close_form(_x, n, L), 'b-')
        l2, = axs[i].plot(_x, pdf_close_form(_x, n, L), 'r-')
        l3, = axs[i].plot(x, psi[:, i], 'bo', markersize=5, alpha=0.8)
        l4, = axs[i].plot(x, pdf[:, i], 'ro', markersize=5, alpha=0.8)

        pdf_analytic = pdf_close_form(x, n, L)
        axs[i].set_title(f'RMSE = {np.round(rmse(pdf_analytic, pdf[:, i]), 5)}')
        axs[i].legend([l1, l2, (l3, l4)], [r'$\psi(x)$', r'$|\psi(x)|^2$', 'FDM'], 
                      handler_map={tuple: HandlerTuple(ndivide=None)})
    plt.xlabel(r'$x$')
    plt.tight_layout()
    plt.show() 

if __name__ == "__main__":
    latexconfig()
    main()