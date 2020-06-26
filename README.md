 # Schrödinger equation solver
 
[![nbviewer](https://raw.githubusercontent.com/jupyter/design/master/logos/Badges/nbviewer_badge.svg)](https://nbviewer.jupyter.org/github/antelk/schrodinger/blob/master/schrodinger.ipynb)
[![Binder](https://mybinder.org/badge_logo.svg)](https://mybinder.org/v2/gh/antelk/schrodinger/169a38fd40f80795b54db12aa441a9602215e681)

[schrodinger.ipynb](https://github.com/antelk/schrodinger/blob/master/schrodinger.ipynb) notebook serves as an official seminar for the graduate course in Modern Physics and Technology [FEMT08](https://nastava.fesb.unist.hr/nastava/predmeti/11624), taught by professor [Ivica Puljak](https://ivicapuljak.com/). The paper titled [Numerical Solution of the Schrödinger Equation Using a Neural Network Approach](https://github.com/antelk/schrodinger/blob/master/1570655320_prereview.pdf) is using this solver and will appear at Advanced Numerical Methods proceedings of SoftCOM2020 conference.

## Installation 

Clone the repo onto your local machine

```bash
$ git clone https://github.com/antelk/schrodinger.git
```

Access `schrodinger` directory

```bash
$ cd schrodinger
```

Create new environment named `schrodinger_ad`

```bash
$ conda env create -f environment.yml -n schrodinger_ad
```

## Use

Activate the environment

```bash
$ conda activate schrodinger_ad
```

Run the notebook

```bash
$ jupyter notebook
```

Deactivate the environment

```bash
$ conda deactivate
```

## Remove from your local machine

Remove the environment and its dependencies

```bash
$ conda remove -n schrodinger_ad --all
```

## Requirements

Library dependencies are listed in [environment.yml](https://github.com/antelk/schrodinger/environment.yml).

## License

[MIT](https://github.com/antelk/schrodinger/LICENSE)