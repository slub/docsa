#!/bin/bash

python setup.py sdist
python setup.py bdist

python -m build --wheel