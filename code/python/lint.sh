#!/bin/bash

export PYTHONPATH="./src"

pylint --max-line-length=120 --extension-pkg-whitelist=lxml ./src/
pylint --max-line-length=120 --extension-pkg-whitelist=lxml ./tests/

flake8 --max-line-length=120 ./src
flake8 --max-line-length=120 ./tests

bandit --configfile bandit.yml -r ./src