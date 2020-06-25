import matplotlib.pyplot as plt
import seaborn as sns
sns.set()
import numpy as np
from scipy.integrate import quad

import sys
sys.path.append('../')
from src.utils.plotting import latexconfig
latexconfig()

def psi(x, n):
    A = np.sqrt(2/L)
    return A * np.sin(n * np.pi / L * x)

def pdf(x, n):
    return psi(x, n)**2

L = 1
quantum_numbers = [1, 2, 3]
x = np.linspace(0, 1, 100)

fig, axs = plt.subplots(3, 1, sharex='all', figsize=(8, 12))
for i, n in enumerate(quantum_numbers):
    axs[i].plot(x, psi(x, n), 'r--', label=r'$\psi(x)$')

    axs[i].plot(x, pdf(x, n), label=r'$|\psi|^2$')
    axs[i].fill_between(x, y1=pdf(x, n), color='lightsteelblue', alpha=0.5)

    I, _ = quad(pdf, 0, L, args=(n))
    axs[i].set_title(f'$n = {n} \Rightarrow \int_{0}^{L}|\psi|^2$dx = {round(I, 2)}')

    axs[i].vlines(x=0, ymin=0, ymax=np.max(pdf(x, n)), color='k')
    axs[i].vlines(x=1, ymin=0, ymax=np.max(pdf(x, n)), color='k')
    axs[i].text(0, np.max(pdf(x,n)), r'$\infty$', color='k')
    axs[i].text(1, np.max(pdf(x,n)), r'$\infty$', color='k')

    if i == 0:
        axs[i].legend(loc='lower center')
    else:
        axs[i].legend(loc='lower left')
    axs[i].set_yticklabels([])
plt.xlabel('$L$ [nm]')
plt.tight_layout()
plt.show()  