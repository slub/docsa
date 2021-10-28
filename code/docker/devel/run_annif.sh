#!/bin/bash

source ./common.sh

${COMPOSE_CMD} -p ${PROJECT_NAME} run --rm --service-ports annif