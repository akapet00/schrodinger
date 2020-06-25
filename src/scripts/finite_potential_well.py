import matplotlib.pyplot as plt
import seaborn as sns
sns.set()
import numpy as np

import sys
sys.path.append('../')
from src.utils.plotting import latexconfig
latexconfig()

# constants 
A = 8.6375   #m/hbar
B = 13.12    #m/hbar**2
hbar = A/B
m = A * hbar

# potential well configuration
x = np.linspace(-0.2, 0.2, 100)
V0 = 1.5 
L = 0.3
E = 0.5

# wave eqn coefs
k_1 = np.sqrt(2*m*E/hbar**2)
k_2 = np.sqrt(2*m*(E+V0)/hbar**2)
A_coeff = 1.0
B_coeff = A_coeff * (np.cos(k_2*L/2)*np.sin(k_1*L/2)-k_2*np.sin(k_2*L/2)*np.cos(k_1*L/2)/k_1)
C_coeff = A_coeff * (np.cos(k_2*L/2)*np.cos(k_1*L/2)+k_2*np.sin(k_2*L/2)*np.sin(k_1*L/2)/k_1)

def psi(x):
    """wave functions for 3 regions"""
    x1 = x[np.where(x<=-L/2)[0]]
    x2 = x[np.where((x>-L/2) & (x<L/2))[0]]
    x3 = x[np.where(x>=L/2)[0]]

    psi1 = -B_coeff * np.sin(k_1*x1) + C_coeff * np.cos(k_1*x1)
    psi2= A_coeff * np.cos(k_2*x2)
    psi3 = B_coeff * np.sin(k_1*x3) + C_coeff * np.cos(k_1*x3)
    return np.r_[psi1, psi2, psi3]

def pdf(x):
    """probability density"""
    return psi(x)**2

fix, ax = plt.subplots(figsize=(8, 4))
# plot the wave function and the probability density
ax.plot(x, psi(x), 'r--', label=r'$\psi(x)$')
ax.plot(x, pdf(x), label=r'$|\psi|^2$')
ax.fill_between(np.linspace(np.min(x), -L/2), 
            pdf(np.linspace(np.min(x), -L/2)),
            color='C0', alpha=0.5)
ax.fill_between(np.linspace(L/2, np.max(x)), 
            pdf(np.linspace(L/2, np.max(x))),
            color='C0', alpha=0.5)

# plot the well w/ potential energy V0
ax.vlines(x=-L/2, ymin=0, ymax=V0, color='k')
ax.vlines(x=L/2, ymin=0, ymax=V0, color='k')
ax.hlines(y=V0, xmin=np.min(x), xmax=-L/2, color='k')
ax.hlines(y=V0, xmin=L/2, xmax=np.max(x), color='k')
ax.text(-L/2, V0+0.03, r'$U_0$')
ax.text(L/2, V0+0.03, r'$U_0$')

labels = [tick.get_text() for tick in ax.get_xticklabels()]
labels[2] = r'$-L/2$'
labels[8] = r'$L/2$'
ax.set_xticklabels(labels)
ax.set_yticklabels([])
plt.legend(loc='upper center')
plt.xlabel('$x$')
plt.show()