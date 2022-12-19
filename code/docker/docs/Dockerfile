FROM docker.io/python:3.8-slim as python

# install make
RUN apt-get update && \
    apt-get install -y make && \
    rm -rf /var/lib/apt/lists/*

# copy all files
COPY . /home/slub/workspace

# install slub_docsa and dependencies
RUN cd /home/slub/workspace && make install-test

# generate documentation
RUN cd /home/slub/workspace/code/python && bash docs.sh

# ------------------------------------------------------------------------

FROM docker.io/nginx:latest

# copy documentation to be served by nginx
COPY --from=python /home/slub/workspace/docs/python /usr/share/nginx/html/
