import matplotlib.pyplot as plt
import seaborn as sns
sns.set()
import numpy as np
from scipy.integrate import quad

def pdf(x, n):
    def Ψ(x, n):
        A = np.sqrt(2/L)
        return A * np.sin(n * np.pi / L * x)
    return Ψ(x, n)**2

L = 1
quantum_numbers = [1, 2, 3]
x = np.linspace(0, 1, 100)

fig, axs = plt.subplots(3, 1, sharex='all', figsize=(6, 8))
for i, n in enumerate(quantum_numbers):
    axs[i].plot(x, pdf(x, n))

    I, _ = quad(pdf, 0, L, args=(n))
    axs[i].set_title(f'$n$ = {n} \t $|\Psi|^2$ = {round(I, 2)}')

    axs[i].vlines(x=0, ymin=0, ymax=np.max(pdf(x, n)), color='r')
    axs[i].vlines(x=1, ymin=0, ymax=np.max(pdf(x, n)), color='r')
    axs[i].text(0, np.max(pdf(x,n)), '$\infty$', color='r')
    axs[i].text(1, np.max(pdf(x,n)), '$\infty$', color='r')


plt.setp(axs, yticks=[]) # yticks removed
plt.xlabel('$L$ [nm]')
plt.tight_layout()
plt.show()  

