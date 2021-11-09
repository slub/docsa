# SLUB Document Classification and Similarity Analysis

This project provides a library for bibliographic document classification and similarity analysis.

It contains a selection of methods that support:

- pre-processing of bibliographic meta data and full-text documents,
- training of multi-label multi-class classification models,
- integrating and using hierarchical subject classifications (pruning methods, performance scores),
- similarity analysis and clustering.

A detailed description including tutorials and examples can be found in the API documentation, which needs to be 
generated as described below.

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

## Documentation

Further documentation of this project can be found at the following locations:

- [API documentation](./docs/python/slub_docsa/index.html) needs to be generated via `make docs` and is than provided 
  in the directory `docs/python/slub_docsa/index.html`.
- Tutorials and examples are described as part of the [API documentation](./docs/python/slub_docsa/index.html)
- Developer meeting notes can be found in a separate 
  [Gitlab Wiki](https://git.slub-dresden.de/lod/maschinelle-klassifizierung/docs/-/wikis/home/Protokolle).
- Results of various experiments related to the [Qucosa](https://www.qucosa.de/) dataset can be found in a separate 
  [Gitlab repository](https://git.slub-dresden.de/lod/maschinelle-klassifizierung/docs/-/tree/main/experiments).

## Development

### Python Virtual Environment

Download all developer dependencies and install the *slub_docsa* package via pip in development mode:

- `make install-dev` 

This will link your local project such that changes to source files are immediately reflected, see 
[pip install -e](https://pip.pypa.io/en/stable/cli/pip_install/#install-editable).

### Container Environment

This project also provides container images for development. You can use [docker](https://docs.docker.com), but also 
other container runtimes, e.g., [podman`](https://podman.io/).

Install a Container Runtime

- Either, install `docker` and `docker-compose`:
  - Install docker, see https://docs.docker.com/get-docker/
  - Install `docker-compose`, see https://docs.docker.com/compose/install/

- Or, setup `podman` in Fedora 34 including the Nvidia container runtime:
  - Install nvidia graphics driver, and check they are working by running `nvidia-smi`
  - Install `podman` and `podman-compose` from repositories via `dnf install podman podman-compose`
  - Install the [nvidia container runtime](https://github.com/NVIDIA/nvidia-container-runtime) using the `centos8`
  repositories via `dnf install nvidia-container-runtime`, see
  [installation instructions](https://docs.nvidia.com/datacenter/cloud-native/container-toolkit/install-guide.html)
  - Set `no-cgroups = true` in `/etc/nvidia-container-runtime/config.toml`, which is required since Nvidia does not yet support cgroups v2
  - Check your CUDA version with `nvidia-smi`, e.g., `11.4`
  - Identify the matching cuda docker image, e.g., `nvidia/cuda:11.4.1-base-centos8`
  - Verify gpu support in podman via
  `podman run --security-opt=label=disable --rm nvidia/cuda:11.4.1-base-centos8 nvidia-smi`

Setup the Development Environment
- Docker images for development can be found in the `code/docker/devel` directory.
- Run `build.sh gpu` to build these docker images with gpu support.
- Run `shell_python.sh gpu` to start a python container with gpu support.
- Run `up.sh gpu` and `shell_annif.sh` to start Annif, `down.sh gpu` to stop. 
  Annif itself does not utilize gpu support though.

Setup [Visual Studio Code](https://code.visualstudio.com/), which supports many useful features during development:
- [Python Integration](https://code.visualstudio.com/docs/languages/python), including auto complete, linting, debugging
- [Remote Container](https://marketplace.visualstudio.com/items?itemName=ms-vscode-remote.remote-containers), which 
  allows to use the Python environment provided by a docker container

Continuous Integration
- The CI pipeline can be triggered by running `make coverage` and `make lint`. Both commands run automated tests using 
[pytest](https://pytest.org/), ensure code guidelines by using [pylint](https://www.pylint.org/) and 
[flake8](https://flake8.pycqa.org/), and check for common security issues using 
[bandit](https://github.com/PyCQA/bandit).
