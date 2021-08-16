#!/bin/bash

(cd src && pylint --max-line-length=120 --extension-pkg-whitelist=lxml *)
(cd tests && pylint --max-line-length=120 --extension-pkg-whitelist=lxml *)

(cd src && flake8 --max-line-length=120)
(cd tests && flake8 --max-line-length=120)

bandit --configfile bandit.yml -r .