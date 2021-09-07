#!/bin/bash

export PYTHONPATH="./src"

coverage run -m --source=./src pytest ./tests
coverage report -m
coverage xml -o ./.coverage.xml