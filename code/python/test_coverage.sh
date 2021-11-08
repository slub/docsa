#!/bin/bash

coverage run -m --source=slub_docsa pytest ./tests
coverage report -m
coverage xml -o ./.coverage.xml