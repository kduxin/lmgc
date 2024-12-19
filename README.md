# Introduction

This repository contains the code for reproducing the results in the paper
"Information-Theoretic Generative Clustering of Documents" by Xin Du and Kumiko Tanaka-Ishii.

- For the paper, please find it at: http://arxiv.org/abs/2412.13534

Generative clustering is a novel clustering paradigm to use generative language
models for document clustering. The language model evaluates the "translation"
probability of each document into a set of short texts (or "queries").
Then, clustering is performed to minimize an information-theoretic distortion
due to the clustering process, defined with these translation probabilities.

Specifically, we solve the following technical difficulites in this paper:
1. Estimate the KL divergence of distributions over the infinite set of word sequences by using importance sampling.
2. Choice of the "optimal" proposal distribution for importance sampling.
3. Stable discovery the true clustering structure through variance control.

Our method achieves the state-of-the-art clustering performance on multiple document clustering datasets.
The following table displays the normalized mutual information scores, a summary of the tested methods.

| | R2 | R5 | AG News | Yahoo! Answers |
|-|-|-|-|-|
| Bag-of-words + kmeans | 3.3 | 20.0 | 1.68 | 3.1 | 
| BERT + kmeans | 31.1 | 27.3 | 38.2 | 9.4 |
| Best of multiple SBERT models + kmeans | 65.6 | 68.9 | 56.7 | 42.8 |
| Best of multiple non-k-means methods | 62.3 | 61.5 | 60.3 | 36.1 |
|-|-|-|-|-|
| GC (ours, a single model) | **77.8** | **71.5** | **64.2** | **43.7** | 


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
maturin develop --release
```
`maturin` is a tool for compiling Rust lib into Python extension.
It should have been installed in the execution of `pip install` above.


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
