#!/bin/bash

(cd ../../../ && podman build -t slub_docsa_production:latest -f code/docker/prod/Dockerfile .)