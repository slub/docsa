#!/bin/bash

podman save slub_docsa_production:latest | gzip > slub_docsa_production.tar.gz