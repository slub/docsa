#!/bin/bash

source ./common.sh

docker-compose -p ${PROJECT_NAME} pull
docker-compose -p ${PROJECT_NAME} build