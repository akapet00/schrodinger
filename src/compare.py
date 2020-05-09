import numpy as np 
import matplotlib.pyplot as plt 
import seaborn as sns
sns.set()

from fdm_inifinite_potential_well import pdf_close_form, fdm
from fem_inifinite_potential_well import fem
from utils.metrics import rmse

def main():
    N = 100 
    xmin = 0.0 
    L = 1.0 
    bcs = [0, 0]    # boundary conditions at x=0 & x=L
    xmesh = np.linspace(xmin, L, N)

    # wave eqn analytical sol for particle in a box problem
    principal_quantum_numbers = list(range(1, 6))
    pdf_analytic = {}
    for n in principal_quantum_numbers:
        pdf_analytic[n] = pdf_close_form(xmesh, n, L)

    # fdm
    _, _, fdm_pdf = fdm(N, xmin, L, bcs) 
    fdm_rmse = {}

    # fem 
    #_, _, fem_pdf = fem(N, xmin, L, bcs)
    fem_rmse = {}

    # nn 
    # placeholder
    ann_rmse = {}
    
    # plotting 
    _, axs = plt.subplots(3, 1, sharex='all', figsize=(7, 9))
    for i in range(3):
        n = i+1     # principal quantum number
        axs[i].plot(xmesh, fdm_pdf[:, i], label=f'FDM')
        #axs[i].plot(xmesh, fem_pdf[:, i], label=f'FEM')
        #axs[i].plot(xmesh, ann_pdf, label=f'ANN')

        axs[i].legend(loc='upper right')
    plt.xlabel(r'x')
    
    # error metrics 
    for n in range(1, 6):
        fdm_rmse[n] = np.round(rmse(pdf_analytic[n], fdm_pdf[:, n-1]), 5)
        #fem_rmse[n] = np.round(rmse(pdf_analytic[n], fem_pdf[:, n-1]), 5)
        #ann_rmse[n] = np.round(rmse(pdf_analytic[n], ann_pdf[:, n-1]), 5)
    print(f'########\n# RMSE #\n########\n\
FDM\n---\n{fdm_rmse}\n\nFEM\n---\n{fem_rmse}\n\nANN\n---\n{ann_rmse}')   

if __name__ == "__main__":
    main()
    plt.show()