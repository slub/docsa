#!/bin/bash

source ./common.sh

docker-compose -p ${PROJECT_NAME} run --rm --service-ports annif