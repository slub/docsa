# SLUB Document Classification and Similarity Analysis (DOCSA)

## Requirements

This project uses Docker to provide a development and test environment.

- Install Docker, see https://docs.docker.com/get-docker/

- Install `docker-compose`, see https://docs.docker.com/compose/install/


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

The CI pipeline can be triggered by building the docker image `code/docker/ci/Dockerfile` (**note**: currently moved to the root directory due to gitlab AutoDevOps issue), which runs automated tests using [pytest](https://pytest.org/), ensures code guidelines by using [pylint](https://www.pylint.org/) and [flake8](https://flake8.pycqa.org/), and checks for common security issues using [bandit](https://github.com/PyCQA/bandit).



