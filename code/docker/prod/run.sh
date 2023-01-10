#!/bin/bash

podman run --rm -it -p 8080:5000 -v ../../../data:/home/slub/workspace/data:Z slub_docsa_production:latest