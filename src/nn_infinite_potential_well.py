import time
import argparse

import autograd.numpy as np 
from scipy.integrate import simps
from scipy.constants import h, hbar, m_e 
import matplotlib.pyplot as plt

from neural_schroedinger.solver import NN

# parser = argparse.ArgumentParser('Infinite potential well demo')
# parser.add_argument()

L = 1
n = 1

def psi(x, n):
    A = np.sqrt(2/L)
    return A * np.sin(n * np.pi / L * x)

def pdf(x, n):
    return psi(x, n)**2

def f(x, y):
    return -(n**2 * h**2 / (8 * m_e))*2*m_e/hbar**2*y

# generate data 
x = np.linspace(0, 1, 50).reshape(-1, 1)
bcs = (0.0, 0.0)
psi_anal = psi(x, n) 
pdf_anal = pdf(x, n) 
I_anal = simps(pdf_anal.ravel(), x.ravel())

sizes = [
    x.shape[1], 
    10,
    10, 
    1]
model = NN(f, x, bcs, sizes=sizes)
print(model)

start = time.time() 
model.fit(maxiter=10000)
end = time.time() 
print(f'\nTraining time: {round((end - start), 4)} s')

psi, _ = model.predict() 
pdf = psi**2 
I = simps(pdf.ravel(), x.ravel())

# integrals
print(f'Integral of analytic solution is {I_anal}')
print(f'Integral of neural solution is {I}')

# plotting 
fig, ax = plt.subplots(nrows=2, ncols=1, sharex=True, squeeze=True)

# analytical
#ax[0].plot(x, psi_anal, 'k--', label=r'$\psi(x)$')
ax[0].plot(x, pdf_anal, 'k-', label=r'$|\psi(x)|^2$')
ax[0].grid()
ax[0].legend()

# neural network
#ax[1].plot(x, psi, 'k--', label=r'$\hat\psi(x)$')
ax[1].plot(x, pdf, 'k-', label=r'$|\hat\psi(x)|^2$')
ax[1].grid()
ax[1].legend()

plt.xlabel(r'$x$')
plt.show()
