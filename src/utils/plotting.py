import numpy as np
import matplotlib as mpl
import matplotlib.pyplot as plt


def latexconfig():
    pgf_latex = {
        # "pgf.texsystem": "pdflatex",
        # "text.usetex": True,
        "font.family": "serif",
        "font.serif": [],
        "font.sans-serif": [],
        "font.monospace": [],
        "axes.labelsize": 10,
        "font.size": 10,
        "legend.fontsize": 10,
        "xtick.labelsize": 10,
        "ytick.labelsize": 10,
        "figure.figsize": figsize(1.0),
        "pgf.preamble": [
            r"\usepackage[utf8x]{inputenc}",
            r"\usepackage[T1]{fontenc}",
            ]
        }
    mpl.rcParams.update(pgf_latex)


def figsize(scale, nplots=1):
    """Golden ratio between the width and height: the ratio 
    is the same as the ratio of their sum to the width of 
    the figure. 
    
    width + height    height
    -------------- = --------
         width        width
    Props for the code goes to:
    https://github.com/maziarraissi/PINNs/blob/master/Utilities/plotting.py
    Parameters
    ----------
    scale : int
        Figure scaler
    nplots : int, optional
        Number of subplots on a single figure
    Returns
    -------
    tuple
        figsize
    """
    fig_width_pt = 390.0
    inches_per_pt = 1.0 / 72.27
    golden_mean = (np.sqrt(5.0) - 1.0) / 2.0
    fig_width = fig_width_pt * inches_per_pt * scale
    fig_height = fig_width * golden_mean * nplots
    return (fig_width, fig_height)