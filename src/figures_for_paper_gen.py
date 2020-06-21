import numpy as np
import matplotlib
import matplotlib.pyplot as plt
font = {
    'family': 'Times New Roman',
    'size': 8}
matplotlib.rc('font', **font)
def figsize(scale, nplots=1):
    fig_width_pt = 390.0                               
    inches_per_pt = 1.0/72.27
    golden_mean = (np.sqrt(5.0)-1.0)/2.0
    fig_width = fig_width_pt*inches_per_pt*scale 
    fig_height = fig_width*golden_mean*nplots
    fig_size = [fig_width,fig_height]
    return fig_size

from fdm_infinite_potential_well import psi_close_form, pdf_close_form, fdm
from fem_infinite_potential_well import fem
from neural_schroedinger import NN
from utils.metrics import rmse

N = 100
xmin = 0.0
L = 1.0
bcs = (0., 0.)
xmesh = np.linspace(xmin, L, N)
x = xmesh.reshape(-1, 1)

psi_analytic = psi_close_form(xmesh, 1, L)
pdf_analytic = pdf_close_form(xmesh, 1, L)
# _, _, pdf_fdm = fdm(N, xmin, L, bcs)
# _, _, pdf_fem = fem(N-1, xmin, L, bcs)
# sizes = [x.shape[1]] + 1 * [40] + [1]
# model = NN(x, bcs, sizes=sizes)
# print(model)
# model.fit(tol=1e-50)
# ann_psi, _ = model.predict() 
# ann_pdf = np.zeros(shape=(N, 1))
# ann_pdf[:, 0] = ann_psi.ravel()**2

# fdm_rmse = np.round(rmse(pdf_analytic, pdf_fdm[:, 0]), 5)
# fem_rmse = np.round(rmse(pdf_analytic, pdf_fem[:, 0]), 5)
# ann_rmse = np.round(rmse(pdf_analytic, ann_pdf[:, 0]), 5)
# print(
#     f'FDM RMSE = {fdm_rmse}',
#     f'FEM RMSE = {fem_rmse}',
#     f'ANN RMSE = {ann_rmse}',
# )

# fig = plt.figure(figsize=figsize(1, 1))
# plt.plot(xmesh, pdf_fdm[:, 0], 'r-', label=r'$|\psi(x)|^2$ FDM')
# plt.plot(xmesh, pdf_fem[:, 0], 'b--', label=r'$|\psi(x)|^2$ FEM')
# plt.plot(xmesh, ann_pdf, 'k-.', label=r'$|\psi(x)|^2$ NN')
# plt.xlabel(r'$x$ [m]')
# plt.ylabel(r'$y$')
# plt.grid()
# plt.legend(loc='best')
# plt.show()
# fig.savefig(f'figs/n1N{N}.pdf', bbox_inches='tight')

fig = plt.figure(figsize=figsize(1, 1))
plt.plot(xmesh, psi_analytic, 'k--', label=r'$\psi(x)$')
plt.plot(xmesh, pdf_analytic, 'k-', label=r'$|\psi(x)|^2$')
plt.xlabel(r'$x$ [m]')
plt.grid()
plt.legend(loc='best')
plt.show()
fig.savefig(f'figs/analytic.pdf', bbox_inches='tight')