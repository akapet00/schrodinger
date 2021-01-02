 # schrodinger
 
[![nbviewer](https://raw.githubusercontent.com/jupyter/design/master/logos/Badges/nbviewer_badge.svg)](https://nbviewer.jupyter.org/github/antelk/schrodinger/blob/master/schrodinger.ipynb)

[schrodinger.ipynb](https://github.com/antelk/schrodinger/blob/master/schrodinger.ipynb) notebook serves as the official seminar for the graduate course in Modern Physics and Technology [FEMT08](https://nastava.fesb.unist.hr/nastava/predmeti/11624), taught by professor [Ivica Puljak](https://ivicapuljak.com/).

The paper titled Numerical Solution of the Schrödinger Equation Using a Neural Network Approach is based on this solver and is available at: https://ieeexplore.ieee.org/document/9238221.


## Cite

```tex
@inProceedings{Kapetanovic2020,
    author={A. L. {Kapetanović} and D. {Poljak}},
    booktitle={2020 International Conference on Software, Telecommunications and Computer Networks (SoftCOM)},
    title={Numerical Solution of the Schrödinger Equation Using a Neural Network Approach},
    year={2020},
    pages={1-5},
    doi={10.23919/SoftCOM50211.2020.9238221}}
```


## Installation 

Clone the repo onto your local machine:
```bash
$ git clone https://github.com/antelk/schrodinger.git
```
Access `schrodinger` directory:
```bash
$ cd schrodinger
```
Create `conda` environment to avoid compatibility issues:
```bash
$ conda env create -n schrodinger -f environment.yml
```
## Use
Activate the environment:
```bash
$ conda activate schrodinger
```
Run:
```bash
$ jupyter notebook
```


## Remove from your local machine

Remove the environment and its dependencies
```bash
$ conda remove -n schrodinger --all
```


## License

[MIT](https://github.com/antelk/schrodinger/LICENSE)