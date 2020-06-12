import matplotlib.pyplot as plt
from matplotlib.widgets import Slider
import seaborn as sns
sns.set()
import numpy as np

import sys
sys.path.append('../')
from src.utils.plotting import latexconfig
latexconfig()

def double_slit_intensity(a, λ, L, d, x) :
    """
    Return intensity for given input parameters.
    
    a (float) - single slit width
    λ (float) - wavelength
    L (float) - screen distance
    d (float) - distance between slits
    x (array) - screen size
    """
    return ((np.sin((np.pi*a*x)/(λ*L))) \
             /((np.pi*a*x)/(λ*L)))**2 \
             *(np.cos((np.pi*d*x)/(λ*L)))**2

x = np.arange(-5e-3, 5e-3, 1e-5)
a = 100e-6
λ = 500e-9
L = 50e-2
d = 1e-3

Y = double_slit_intensity(a, λ, L, d, x)
plt.figure('Double-slit Experiment', figsize=(8,4))
p,  = plt.plot(x, Y)
plt.xlabel('x [m]')
plt.ylabel('Intensity')

s1 = (plt.axes([.68, .8, .14, .05]))
s2 = (plt.axes([.68, .7, .14, .05]))
s3 = (plt.axes([.68, .6, .14, .05]))
s4 = (plt.axes([.68, .5, .14, .05]))


a_slider = Slider(s1, 'a [$\mu$m]', 10, 1000, valinit=a*1e6)
λ_slider = Slider(s2, '$\lambda$ [nm]', 100, 1000, valinit=λ*1e9)
L_slider = Slider(s3, 'L [cm]', 10, 100, valinit= L*1e2)
d_slider = Slider(s4, 'd [mm]', 0.1, 2, valinit=d*1e3)

def update(val) :
    a = a_slider.val*1e-6
    λ = λ_slider.val*1e-9
    L = L_slider.val*1e-2
    d = d_slider.val*1e-3
    Y = double_slit_intensity(a, λ, L, d, x)
    p.set_ydata(Y)

a_slider.on_changed(update)
λ_slider.on_changed(update)
L_slider.on_changed(update)
d_slider.on_changed(update)

plt.show()