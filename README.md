# Introduction

This repository contains the code for reproducing the results in the paper
``Information-Theoretic Generative Clustering of Documents'' by Xin Du and Kumiko Tanaka-Ishii.


# Environment

The code is written in Python 3.10.

You may want to work in a virtual environment. We used [pyenv](https://github.com/pyenv/pyenv) to manage Python versions and [pyenv-virtualenv](https://github.com/pyenv/pyenv-virtualenv) to manage virtual environments. Install both, then create a new virtual environment by running
```bash
$ pyenv virtualenv 3.10.0 lmgenc
$ pyenv shell lmgenc
```

Install the required packages by running
```bash
pip install -r requirements.txt
```

Compile our extension module by running
```bash
python setup.py build_ext --inplace
```


# Quick start

To reproduce the results for the R2 dataset, run
```bash
$ bash scripts/prepare.sh
```
which would takes around 2 minutes on 8 x RTX4090 GPUs.

Then, execute clustering by running
```bash
$ bash scripts/cluster.sh
```
