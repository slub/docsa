# SLUB Document Classification and Similarity Analysis (DOCSA)

## Requirements

This project uses Docker to provide a development and test environment.

- Install Docker, see https://docs.docker.com/get-docker/

- Install `docker-compose`, see https://docs.docker.com/compose/install/

### Setup Nvidia with Podman in Fedora 34

- Install nvidia graphics driver, and check they are working by running `nvidia-smi`

- Install podman from repositories via `dnf install podman`

- Install nvidia-container-runtime using `centos8` repositories, see via `dnf install nvidia-container-runtime`

- Set `no-cgroups = true` in `/etc/nvidia-container-runtime/config.toml`

- Check your CUDA version with `nvidia-smi`, e.g., `11.4`

- Identify the matching cuda docker image, e.g., `nvidia/cuda:11.4.1-base-centos8`

- Run to verify gpu support in podman via `podman run --security-opt=label=disable --rm nvidia/cuda:11.4.1-base-centos8 nvidia-smi`
## Development

Docker images for development can be found in the `code/docker/devel` directory.

- Run `build.sh` to build these docker images.

- Run `shell_python.sh` to run a python container

- Run `up.sh` and `shell_annif.sh` to start Annif, `down.sh` to stop


## Visual Studio Code

[Visual Studio Code](https://code.visualstudio.com/) supports many useful features during development:

- [Python Integration](https://code.visualstudio.com/docs/languages/python), including auto complete, linting, debugging

- [Remote Container](https://marketplace.visualstudio.com/items?itemName=ms-vscode-remote.remote-containers), which allows to use the Python environment provided by a docker container


## Continuous Integration

The CI pipeline can be triggered by building the docker image `code/docker/ci/Dockerfile`, which runs automated tests using [pytest](https://pytest.org/), ensures code guidelines by using [pylint](https://www.pylint.org/) and [flake8](https://flake8.pycqa.org/), and checks for common security issues using [bandit](https://github.com/PyCQA/bandit).



