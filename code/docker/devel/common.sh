GPU_MODE=$1

export DOCKER_CMD="podman"
# export DOCKER_CMD="docker"
export COMPOSE_CMD="podman-compose"
# export COMPOSE_CMD="docker-compose"

if [ "$GPU_MODE" == "gpu" ]; then
    TENSORFLOW_IMAGE_TAG="2.6.0-gpu"
fi

if [ "$GPU_MODE" == "cpu" ]; then
    TENSORFLOW_IMAGE_TAG="2.6.0"
fi

if [ -z "$TENSORFLOW_IMAGE_TAG" ]; then
    echo "You need to provide build mode as argument: either 'cpu' or 'gpu'."
    exit 1
fi

export TENSORFLOW_IMAGE_TAG=${TENSORFLOW_IMAGE_TAG}
export PROJECT_NAME="slub_docsa_${GPU_MODE}"