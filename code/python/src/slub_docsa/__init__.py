"""
This python package is a library for bibliographic document classification and similarity analysis.

It provides a selection of methods that support:

- pre-processing of bibliographic meta data and full-text documents,
- training of multi-label multi-class classification models,
- integrating and using hierarchical subject classifications (pruning methods, performance scores),
- similarity analysis and clustering.

Some important features include:

- a concise API for training and evaluating multi-label multi-class classification models, see `slub_docsa.common`
- support for many different classification approaches, see `slub_docsa.models`
- artificial hierarchical random datasets, see `slub_docsa.data.artificial`
- performance scores that consider hierarchical relations, see `slub_docsa.evaluation.score`

## Installation

This projects requires [Python](https://www.python.org/) v3.6 or above and uses [pip](https://pypi.org/project/pip/)
for dependency management. Besides, this package uses [pyTorch](https://pytorch.org/) to train
[Artificial Neural Networks](https://en.wikipedia.org/wiki/Artificial_neural_network) via GPUs.
Make sure to install the latest Nvidia graphics drivers and check
[further requirements](https://pytorch.org/get-started/locally/#linux-prerequisites).

### Via Python Package Installer (not available yet)

Once published to PyPI (*not available yet*), install via:

- `python3 -m pip install slub_docsa`

### From Source

Download the source code by checking out the repository:

 - `git clone https://git.slub-dresden.de/lod/maschinelle-klassifizierung/docsa.git`

Use *make* to install python dependencies by executing the following commands:

- `make install` or `make install-test`
  (installs *slub_docsa* package and downloads all required runtime / test dependencies via *pip*)
- `make test`
  (runs tests to verify correct installation, requires test dependencies)
- `make docs`
  (generate API documentation, requires test dependencies)

### From Source using Ubuntu 18.04

Install essentials like *python3*, *pip* and *make*:

- `apt-get update`
   (update the Ubuntu package installer index)
- `apt-get install -y make python3 python3-pip`
   (install python3, pip and make)

Optionally, set up a python [virtual environment](https://docs.python.org/3/tutorial/venv.html):

- `apt-get install -y python3-venv`
- `python3 -m venv /path/to/venv`
- `source /path/to/venv/bin/activate`

Run *make* commands as provided above:

- `make install-test`
- `make test`

## Configuration

--TODO--

## First Steps

--TODO--

## Examples

--TODO--

"""
