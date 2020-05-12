 # Schrödinger equation solver
 
[![nbviewer](https://raw.githubusercontent.com/jupyter/design/master/logos/Badges/nbviewer_badge.svg)](https://nbviewer.jupyter.org/github/antelk/schrodinger/blob/master/schrodinger.ipynb)
[![Binder](https://mybinder.org/badge_logo.svg)](https://mybinder.org/v2/gh/antelk/schrodinger/169a38fd40f80795b54db12aa441a9602215e681)

[schrodinger.ipynb](https://github.com/antelk/schrodinger/blob/master/schrodinger.ipynb) notebook serves as an official seminar for the graduate course in Modern Physics and Technology [FEMT08](https://nastava.fesb.unist.hr/nastava/predmeti/11624), taught by professor [Ivica Puljak](https://ivicapuljak.com/).

## Installation 

Clone the repo onto your local machine

```bash
$ git clone https://github.com/antelk/schrodinger.git
```

Access `schrodinger` directory

```bash
$ cd schrodinger
```

Create new environment named `schrodinger_tf2_gpu`

```bash
$ conda env create -f environment.yml -n schrodinger_tf2_gpu
```

## Use

Activate the environment

```bash
$ conda activate schrodinger_tf2_gpu
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
$ conda remove -n schrodinger_tf2_gpu --all
```

## Requirements

Linux machine with Nvidia GPU support, for more details visit [TensorFlow official hardware & software requirements](https://www.tensorflow.org/install/gpu).

Library dependencies are listed in [environment.yml](https://github.com/antelk/schrodinger/environment.yml).

## License

[MIT](https://github.com/antelk/schrodinger/LICENSE)