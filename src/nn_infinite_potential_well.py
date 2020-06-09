from neural_schroedinger import NN

import argparse
import autograd.numpy as np 
from scipy.integrate import simps, quad
from scipy.constants import h, hbar, m_e 
import matplotlib.pyplot as plt

parser = argparse.ArgumentParser('Infinite potential well demo')
parser.add_argument('-x', '--training_points', type=int, default=50,
    help='Number of grid points over the x-axis that will serve\
         as training points.')
parser.add_argument('-L', '--domain_length', type=float, default=1e0,
    help='Solution domain from 0 to L over x-axis.')
parser.add_argument('-l', '--hidden_layers', type=int, default=1,
    help='Number of hidden layers.')
parser.add_argument('-n', '--hidden_units', type=int, default=16,
    help='Number of hidden units per hidden layer.')
parser.add_argument('-a', '--activation', type=str, default='tanh',
    choices=['tanh', 'sigmoid', 'relu', 'elu', 'softplus', 'prelu'], 
    help='Activation function for both input and hidden layers.')
parser.add_argument('-o', '--optimizer', type=str, default='bfgs',
    choices=['BFGS', 'L-BFGS-B'], 
    help='Algorithm for the minimization of loss function.')
parser.add_argument('-i', '--iteration', type=int, default=2000,
    help='Number of training iterations for optimizer.')
parser.add_argument('-t', '--tolerance', type=float, default=1e-20,
    help='Optimizer threshold value.')
parser.add_argument('-q', '--quantum_state', type=int, default=1,
    help='Principal quantum number - 1, 2, 3, ...')
args = parser.parse_args()

# generate training data 
x = np.linspace(0, args.domain_length, args.training_points).reshape(-1, 1)
bcs = (0.0, 0.0)

# analytical functions
psi_anal = lambda x, n, L : np.sqrt(2/L) * np.sin(n * np.pi / L * x)
pdf_anal = lambda x, n, L: psi_anal(x, n, L)**2
psi_anal_sampled = psi_anal(x, args.quantum_state, args.domain_length) 
pdf_anal_sampled = pdf_anal(x, args.quantum_state, args.domain_length) 
I_anal, _ = quad(pdf_anal, min(x), max(x), args=(args.quantum_state, args.domain_length))

# neural network architecture and training
sizes = [x.shape[1]] + args.hidden_layers * [args.hidden_units] + [1]
model = NN(x, bcs, sizes=sizes, activation=args.activation)
print(model) 
model.fit(method=args.optimizer, maxiter=args.iteration, tol=args.tolerance)
model.plot_loss(log=False)
psi, _ = model.predict() 
pdf = psi**2 
I = simps(pdf.ravel(), x.ravel())

# print integrals to console
print(f'Integral of analytic solution is {I_anal}')
print(f'Integral of neural solution is {I}')

# plotting 
fig, ax = plt.subplots(nrows=2, ncols=1, sharex=True, squeeze=True)
# analytical
ax[0].plot(x, psi_anal_sampled, 'k--', label=r'$\psi(x)$')
ax[0].plot(x, pdf_anal_sampled, 'k-', label=r'$|\psi(x)|^2$')
ax[0].grid()
ax[0].legend()
# neural network
ax[1].plot(x, psi, 'k--', label=r'$\hat\psi(x)$')
ax[1].plot(x, pdf, 'k-', label=r'$|\hat\psi(x)|^2$')
ax[1].grid()
ax[1].legend()
plt.xlabel(r'$x$')
plt.show()