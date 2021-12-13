#!/bin/bash

# remove any existing installed packages
python3 -m pip uninstall -y slub_docsa

# build package
bash ./package_build.sh

# install built wheel file
python3 -m pip install dist/*.whl

# run tests on production build
bash ./test_run.sh

# reinstall developer version
python3 -m pip uninstall -y slub_docsa
python3 -m pip install -e .[dev,test]