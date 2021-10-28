#!/bin/bash

source ./common.sh

${COMPOSE_CMD} -p ${PROJECT_NAME} pull
${COMPOSE_CMD} -p ${PROJECT_NAME} build