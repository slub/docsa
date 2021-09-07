#!/bin/bash

(cd ./src && pylint --max-line-length=120 --extension-pkg-whitelist=lxml *)
(cd ./src && pylint --max-line-length=120 --extension-pkg-whitelist=lxml ../tests)

(cd ./src && flake8 --max-line-length=120)
(cd ./src && flake8 --max-line-length=120 ../tests)

bandit --configfile bandit.yml -r .