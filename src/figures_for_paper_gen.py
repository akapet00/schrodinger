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

# do not change this
xmin = 0.0
L = 1.0
bcs = (0., 0.)

############
# wave eqn #
############
# N = 100
# xmesh = np.linspace(xmin, L, N)
# x = xmesh.reshape(-1, 1)
# psi_analytic = psi_close_form(xmesh, 1, L)
# pdf_analytic = pdf_close_form(xmesh, 1, L)
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

# fig = plt.figure(figsize=figsize(1, 1))
# plt.plot(xmesh, psi_analytic, 'k--', label=r'$\psi(x)$')
# plt.plot(xmesh, pdf_analytic, 'k-', label=r'$|\psi(x)|^2$')
# plt.xlabel(r'$x$ [m]')
# plt.grid()
# plt.legend(loc='best')
# plt.show()
# fig.savefig(f'figs/analytic.pdf', bbox_inches='tight')


#############
# benchmark #
#############
# fdm_rmse = []
# fem_rmse = []
# ann_rmse = []
# N = range(11, 100, 10)
# for n in N:
#     xmesh = np.linspace(xmin, L, n)
#     x = xmesh.reshape(-1, 1)
#     # analytic
#     psi_analytic = psi_close_form(xmesh, 1, L)
#     pdf_analytic = pdf_close_form(xmesh, 1, L)
#     # finite diff
#     _, _, pdf_fdm = fdm(n, xmin, L, bcs)
#     fdm_rmse.append(np.round(rmse(pdf_analytic, pdf_fdm[:, 0]), 5))
#     # finite elem
#     _, _, pdf_fem = fem(n-1, xmin, L, bcs)
#     fem_rmse.append(np.round(rmse(pdf_analytic, pdf_fem[:, 0]), 5))
#     # neural net
#     sizes = [x.shape[1]] + 3 * [40] + [1]
#     model = NN(x, bcs, sizes=sizes)
#     model.fit(tol=1e-50)
#     psi_ann, _ = model.predict() 
#     pdf_ann = np.zeros(shape=(n, 1))
#     pdf_ann[:, 0] = psi_ann.ravel()**2
#     ann_rmse.append(np.round(rmse(pdf_analytic, pdf_ann[:, 0]), 5))

# N = list(N)
# plt.plot(N, ann_rmse, 'k-', label='ANN RMSE')
# plt.plot(N, fem_rmse, 'k--', label='FEM RMSE')
# plt.plot(N, fdm_rmse, 'k-.', label='FDM RMSE')
# plt.legend()
# plt.grid()
# plt.show()

###################
# paper benchmark #
###################
N = list(range(11, 100, 10))
rmse_fdm = [0.11677, 0.05976, 0.04016, 0.03024, 0.02425, 0.02024, 0.01777, 0.01521, 0.01353]
rmse_fem = [0.01937, 0.00493, 0.00220, 0.00124, 0.00080, 0.00056, 0.00041, 0.00031, 0.00025]
rmse_ann = [0.08630, 0.07311, 0.06303, 0.07792, 0.05863, 0.06599, 0.06452, 0.05471, 0.03783]
plt.figure(figsize=figsize(1,1))
plt.plot(N, rmse_fdm, 'r-', label='FDM RMSE')
plt.plot(N, rmse_fem, 'b--', label='FEM RMSE')
plt.plot(N, rmse_ann, 'k-.', label='ANN RMSE')
plt.xlabel(r'the number of collocation points, $N$')
plt.ylabel('RMSE')
plt.legend()
plt.grid()
plt.show()