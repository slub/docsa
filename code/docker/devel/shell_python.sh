#!/bin/bash

source ./common.sh

docker-compose -p ${PROJECT_NAME} run --rm --name ${PROJECT_NAME}_python_shell_1 python
