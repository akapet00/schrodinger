 # Schrödinger equation solver
 
[![Binder](https://mybinder.org/badge_logo.svg)](https://blank.org/)

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

Create new environment named as `schrodinger_tf2_gpu`

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

Activate the environment

```bash
$ conda deactivate
```

## Remove from your local machine

Remove environment and its dependencies

```bash
$ conda remove -n schrodinger_tf2_gpu --all
```

## Requirements

Listed in [environment.yml](https://github.com/antelk/schrodinger/environment.yml).

## License

[MIT](https://github.com/antelk/schrodinger/LICENSE)
