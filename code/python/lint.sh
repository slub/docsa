#!/bin/bash

export PYTHONPATH="./src"

pylint --max-line-length=120 --min-similarity-lines=8 --extension-pkg-whitelist=lxml ./src/
pylint --max-line-length=120 --min-similarity-lines=8 --extension-pkg-whitelist=lxml ./tests/

flake8 --ignore=D301,W503 --max-line-length=120 ./src
flake8 --ignore=D301,W503 --max-line-length=120 ./tests

bandit -c .bandit.yaml -r ./src