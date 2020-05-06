import argparse
import numpy as np
from scipy.constants import h, hbar, m_e
import matplotlib.pyplot as plt
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

def rmse(true, predict):
    return np.sqrt(np.mean((true - predict)**2))

def p(x):
    """source function"""
    return .0 * x

def fdm(N, xmin, L, bcs, args):
    """finite difference
       (A/dx**2 - B/(2*dx)) * psi(x-dx) + (C-2*A/dx**2) * psi(x) + (A/dx**2 + B/(2*dx)) * psi(x+dx)
    """
    dx = (L - xmin)/(N - 1)     # distance between grid points

    A, B, C = args              # eqn arguments
    psi_0, psi_L = bcs          # boundary conditions

    a = A/dx**2 - B/(2*dx)      
    b = C-2*A/dx**2
    c = a - B/(2*dx)

    # a * psi[i-1] + b * psi[i] + c * psi[i+1] = 0
    # LHS @ sol = RHS + bcs 
    lhs = np.diag(np.full(N-3, a), k=-1) +\
          np.diag(np.full(N-2, b)) +\
          np.diag(np.full(N-3, c), k=1)
    LHS = np.zeros((N-1, N-1))
    LHS[1:, 1:] = lhs
    LHS[0,0] = 1/dx * (-1)
    LHS[0,1] = 1/dx
    
    RHS = p(np.linspace(xmin, L, N)).reshape(-1, 1)
    
    sol = np.linalg.solve(LHS, RHS)
    sol[0, ] = psi_0
    sol[-1, ] = psi_L
    
    return sol

def main():
    args = parser.parse_args()
    N = args.num_of_points                      # number of grid points

    # domain configuration
    xmin = .0                                   # left boundary
    L = 1.0                                     # right boundary

    # boundary conditions
    psi_0 = .0
    psi_L = .0

    # kinetic energy of an electron 
    principal_quantum_numbers = [1, 2, 3]       
    nonreletivistic_energies = []
    for n in principal_quantum_numbers:
        E = n**2 * h / (8 * m_e * L)
        nonreletivistic_energies.append(E)

    sol = []
    for n, E in zip(principal_quantum_numbers, nonreletivistic_energies):
        # hbar/(2*m_e) * psi_xx + E * psi = 0 
        A = hbar/(2*m_e)
        B = 0 
        C = E

        sol.append(fdm(N, xmin, L, [psi_0, psi_L], [A, B, C]))

    print(sol)
   
if __name__ == "__main__":
    main()




