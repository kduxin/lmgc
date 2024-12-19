# Introduction

This repository contains the code for reproducing the results in the paper
"Information-Theoretic Generative Clustering of Documents" by Xin Du and Kumiko Tanaka-Ishii.
- For the paper with appendix included, please find it at: http://arxiv.org/abs/2412.13534

Generative clustering is a novel clustering paradigm to use generative language
models for document clustering. The language model evaluates the "translation"
probability of each document into a set of short texts (or "queries").
Then, clustering is performed to minimize an information-theoretic distortion
due to the clustering process, defined with these translation probabilities.

Specifically, we solve the following technical difficulites in this paper:
1. Estimation of the KL divergence over the infinite set of word sequences by using importance sampling.
2. Choice of the "optimal" proposal distribution for importance sampling. In other words, how to select the short texts.
3. Stable discovery the true clustering structure through variance control.

Our method achieves the state-of-the-art clustering performance on multiple document clustering datasets.
The following table displays the normalized mutual information scores, a summary of the tested methods.
For conciseness, for a class of similar methods, we report the best score among them,
while for our GC (the bottom row) the scores are achieved by a simple model.

| | R2 | R5 | AG News | Yahoo! Answers |
|-|-|-|-|-|
| Bag-of-words + kmeans | 3.3 | 20.0 | 1.68 | 3.1 | 
| BERT + kmeans | 31.1 | 27.3 | 38.2 | 9.4 |
| Best of multiple SBERT models + kmeans | 65.6 | 68.9 | 56.7 | 42.8 |
| Best of multiple non-k-means methods | 62.3 | 61.5 | 60.3 | 36.1 |
|-|-|-|-|-|
| GC (ours, a single model) | **77.8** | **71.5** | **64.2** | **43.7** | 

Other advantages of this new method:
- Interpretability: It represents documents in a low-dimensional space, where each dimension is a text. So each dimension has a concrete meaning.
- Low but extensible dimensionality (each dim is an iid sample): requires fewer dimensions than BERT-like methods. On R2, using 30 dimensions outperforms SBERT.
- As simple as k-means.

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
which would takes no more than 20 minutes on a single GPU card.

Then, execute clustering by running
```bash
$ bash scripts/cluster.sh
```
