#!/bin/bash

cd ../../../
docker build -t slub_docsa_python_ci -f code/docker/ci/Dockerfile .