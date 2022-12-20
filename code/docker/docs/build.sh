#!/bin/bash

cd ../../../
podman build -t slub_docsa_docs:latest -f code/docker/docs/Dockerfile .