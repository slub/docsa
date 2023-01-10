FROM docker.io/python:3.8-slim

# install development basics
RUN apt-get update && \
    apt-get install -y git vim screen wget make python3-venv && \
    rm -rf /var/lib/apt/lists/*

# setup python virtual environment
RUN mkdir -p /home/slub
RUN python3 -m venv /home/slub/.venv
ENV PATH="/home/slub/.venv/bin:$PATH"

# disable caching in pip for smaller image size
ARG PIP_DISABLE_PIP_VERSION_CHECK=1
ARG PIP_NO_CACHE_DIR=1

# copy code and install runtime requirements
COPY . /home/slub/workspace
RUN cd /home/slub/workspace && make install-dev

# declare data directory volume
VOLUME ["home/slub/workspace/data"]

# run rest service
WORKDIR /home/slub/workspace
CMD ["make", "serve"]
