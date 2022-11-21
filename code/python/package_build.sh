#!/bin/bash

# remove and previously build wheel files
rm -rf dist

python setup.py sdist
python setup.py bdist

python -m build --wheel