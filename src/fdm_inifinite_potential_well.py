import argparse
import numpy as np
from scipy.constants import h, hbar, m_e, pi
import matplotlib.pyplot as plt
from matplotlib.legend_handler import HandlerTuple
import seaborn as sns
sns.set()

parser = argparse.ArgumentParser(description='Takes the number of grid points.')
parser.add_argument('-n', '--num_of_points',
                    type=int,
                    metavar='',
                    help='Finite Difference approximation of the wave equation for\
                        an electron in the infinite quantum well based on the given\
                        number of grid points.',
                    default=10)
                             
def mape(true, predict):
    """mean absolute percentage errror"""
    return np.mean(np.abs(true - predict)/(1 + np.abs(true))) * 100.0

def rmse(true, predict):
    """root mean squared error"""
    return np.sqrt(np.mean((true-predict)**2))

def psi_close_form(x, n, L):
    """analyitcal solution for 1-D wave eqn"""
    amp = np.sqrt(2/L)
    return amp * np.sin(n * pi / L * x)

def pdf_close_form(x, n, L):
    r"""probability of locating an electron in x \in (0, L)"""
    return psi_close_form(x, n, L)**2

def p(x):
    """source function"""
    return 0. * x

def fdm(N, xmin, L, bcs, args):
    """(A/dx**2 - B/(2*dx)) * psi(x-dx) + (C-2*A/dx**2) * psi(x) + (A/dx**2 + B/(2*dx)) * psi(x+dx)"""
    dx = (L - xmin)/(N - 1)         # distance between grid points
    xmesh = np.linspace(xmin, L, N) # mesh

    A, B, C = args                  # eqn arguments
    psi_0, psi_L = bcs              # boundary conditions

    a = A/dx**2 - B/(2*dx)      
    b = C-2*A/dx**2
    c = A/dx**2  + B/(2*dx)

    # a * psi[i-1] + b * psi[i] + c * psi[i+1] = 0
    # LHS @ sol = RHS + bcs 
    LHS = np.diag(np.full(N, a), k=0) +\
         np.diag(np.full(N-1, b), k=-1) + np.diag(np.full(N-1, b), k=+1) +\
         np.diag(np.full(N-2, c), k=2) 
    
    RHS = p(xmesh).reshape(-1, 1)
    
    sol = np.linalg.solve(LHS, RHS)
    sol[0, ] = psi_0
    sol[-1, ] = psi_L
    
    return sol

def main():
    args = parser.parse_args()
    N = args.num_of_points          # number of grid points
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

    fig, axs = plt.subplots(len(principal_quantum_numbers), 1, sharex='all', figsize=(8, 12))
    for i, (n, E) in enumerate(zip(principal_quantum_numbers, nonreletivistic_energies)):
        # hbar**2/(2*m_e) * psi_xx + E * psi = 0 
        A = hbar**2/(2*m_e)
        B = 0 
        C = E
        psi = fdm(N, xmin, L, (psi_0, psi_L), (A, B, C)).flatten()
        pdf = psi**2

        psi_analytic = psi_close_form(x, n, L)
        pdf_analytic = pdf_close_form(x, n, L)
    
        l1, = axs[i].plot(_x, psi_close_form(_x, n, L), 'b-')
        l2, = axs[i].plot(_x, pdf_close_form(_x, n, L), 'r--')
        l3, = axs[i].plot(x, psi, 'bo', markersize=7)
        l4, = axs[i].plot(x, psi**2, 'r^', markersize=7)

        axs[i].set_title(f'RMSE = {round(rmse(pdf_analytic, pdf), 3)}')
        axs[i].legend([l1, l2, (l3, l4)], [r'$\psi(x)$', r'$|\psi(x)|^2$', 'FDM'], 
                      handler_map={tuple: HandlerTuple(ndivide=None)})
    
    plt.xlabel(r'$x$')
    plt.tight_layout()
    plt.show()  


if __name__ == "__main__":
    main()




