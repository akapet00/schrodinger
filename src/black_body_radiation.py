import matplotlib.pyplot as plt
import seaborn as sns
sns.set()
import numpy as np

h = 6.626e-34    # Planck constant [J⋅s]
c = 3.0e+8       # speed of light [m/s]
k = 1.38e-23     # Boltzmann constant [J⋅K^−1]

def B_rj(λ, T):
    """Rayleigh-Jeans black-body radiation interpretation"""
    return (2.0 * c * k * T) / (λ**4)

def B_p(λ, T):
    """Planck black-body radiation formula"""
    return (2.0 * h * c**2) / (λ**5 * (np.exp(h*c/(λ*k*T)) - 1.0))

λ = np.arange(1e-9, 3e-6, 1e-9, dtype=np.float128) 
Ts = [4000., 5000., 6000., 7000.]

plt.figure('Black-body Radiation', figsize=(8, 4))
for T in Ts:
    plt.plot(λ*1e9, B_p(λ, T), label=f'$T={int(T)}K$')

plt.plot(λ*1e9, B_rj(λ, T=5000), label=f'Rayleigh-Jeans (T={5000}K)', color='black')

plt.vlines(x=380, ymin=0, ymax=7e13, color='black', linestyle=':')
plt.vlines(x=740, ymin=0, ymax=7e13, color='black', linestyle=':', 
           label='visible light borders')

#visible spectrum coloring
plt.axvspan(380, 450, alpha=0.3, color='violet')
plt.axvspan(451, 485, alpha=0.3, color='blue')
plt.axvspan(486, 500, alpha=0.3, color='cyan')
plt.axvspan(501, 565, alpha=0.3, color='green')
plt.axvspan(566, 590, alpha=0.3, color='yellow')
plt.axvspan(591, 625, alpha=0.3, color='orange')
plt.axvspan(626, 740, alpha=0.3, color='red')

plt.ylim([0, 7e13])
plt.legend(loc='best')
plt.xlabel('$\lambda$ [nm]')
plt.ylabel('$B(\lambda, T)$ [W·sr−1·m−3]')
plt.show()