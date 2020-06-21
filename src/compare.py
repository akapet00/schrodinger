import argparse
import numpy as np 
import matplotlib.pyplot as plt 
import seaborn as sns
sns.set()

from fdm_infinite_potential_well import pdf_close_form, fdm
from fem_infinite_potential_well import fem
from neural_schroedinger import NN
from utils.metrics import rmse

parser = argparse.ArgumentParser(description='Takes the number of grid points in the 1-D mesh.')
parser.add_argument('-n', '--grid_points',
                    type=int,
                    metavar='',
                    default=10)

def main():
    args = parser.parse_args()
    N = args.grid_points
    xmin = 0.0
    L = 1.0 
    bcs = (0., 0.)
    xmesh = np.linspace(xmin, L, N)

    # wave eqn analytical sol for particle in a box problem
    principal_quantum_numbers = 1
    pdf_analytic = pdf_close_form(xmesh, principal_quantum_numbers, L)

    # fdm
    _, _, fdm_pdf = fdm(N, xmin, L, bcs)

    # fem 
    _, _, fem_pdf = fem(N-1, xmin, L, bcs)

    # nn 
    x = xmesh.reshape(-1,1)
    sizes = [x.shape[1]] + 1 * [40] + [1]
    model = NN(x, bcs, sizes=sizes)
    print(model)
    model.fit(tol=1e-80)
    ann_psi, _ = model.predict() 
    ann_pdf = np.zeros(shape=(N, 1))
    ann_pdf[:, 0] = ann_psi.ravel()**2
    
    # plotting 
    fig, ax = plt.subplots(1, 1, sharex='all')
    ax.plot(xmesh, fdm_pdf[:, 0], 'b-', linewidth=2, label=f'FDM')
    ax.plot(xmesh, fem_pdf[:, 0], 'r-', linewidth=2, label=f'FEM')
    ax.plot(xmesh, ann_pdf[:, 0], 'g-', linewidth=2, label=f'ANN')
    ax.legend(loc='upper right')
    plt.xlabel(r'x')

if __name__ == "__main__":
    main()
    plt.show()