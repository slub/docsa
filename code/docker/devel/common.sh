GPU_MODE=$1

export DOCKER_CMD="podman"
# export DOCKER_CMD="docker"
export COMPOSE_CMD="podman-compose"
# export COMPOSE_CMD="docker-compose"

if [ "$GPU_MODE" == "gpu" ]; then
    BASE_IMAGE="docker.io/nvidia/cuda:11.7.1-cudnn8-runtime-ubuntu20.04"
fi

if [ "$GPU_MODE" == "cpu" ]; then
    BASE_IMAGE="docker.io/python:3.8-slim"
fi

if [ -z "$BASE_IMAGE" ]; then
    echo "You need to provide build mode as argument: either 'cpu' or 'gpu'."
    exit 1
fi

export BASE_IMAGE=${BASE_IMAGE}
export PROJECT_NAME="slub_docsa_${GPU_MODE}"