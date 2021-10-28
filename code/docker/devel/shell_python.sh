#!/bin/bash

source ./common.sh

${COMPOSE_CMD} -p ${PROJECT_NAME} run --rm --name ${PROJECT_NAME}_python_shell_1 python
