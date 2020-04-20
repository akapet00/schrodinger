# -*- coding: utf-8 -*-

import numpy as np
import matplotlib as mpl
import matplotlib.pyplot as plt
    
def latexconfig():
    pgf_latex = {                                       # setup matplotlib to use latex for output
        "pgf.texsystem": "pdflatex",                    # change this if using xetex or lautex
        "text.usetex": True,                            # use LaTeX to write all text
        "font.family": "serif",                         # default LaTex option
        "font.serif": [],                               # blank entries should cause plots to inherit fonts from the document
        "font.sans-serif": [],
        "font.monospace": [],
        "axes.labelsize": 10,                           # LaTeX default is 10pt font
        "font.size": 10,                                # LaTeX default is 10pt font
        "legend.fontsize": 10,                          # Make the legend/label fonts a little smaller
        "xtick.labelsize": 10,
        "ytick.labelsize": 10,
        "figure.figsize": figsize(1.0),                 # default fig size of 0.9 textwidth
        "pgf.preamble": [
            r"\usepackage[utf8x]{inputenc}",            # utf8 input support
            r"\usepackage[T1]{fontenc}",                # plots will be generated using this preamble
            ]
        }
    mpl.rcParams.update(pgf_latex)

def figsize(scale, nplots=1):
    fig_width_pt = 390.0                               # Get this from LaTeX using \the\textwidth
    inches_per_pt = 1.0/72.27                          # Convert pt to inch
    golden_mean = (np.sqrt(5.0)-1.0)/2.0               # Aesthetic ratio (you could change this)
    fig_width = fig_width_pt*inches_per_pt*scale       # width in inches
    fig_height = fig_width*golden_mean*nplots          # height in inches
    fig_size = [fig_width,fig_height]
    return fig_size