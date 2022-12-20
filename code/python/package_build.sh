#!/bin/bash

# remove and previously build and wheel files
rm -rf build
rm -rf dist


python setup.py sdist
python setup.py bdist

python -m build --wheel