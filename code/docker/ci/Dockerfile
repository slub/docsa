FROM python:3.6-slim

# install make
RUN apt-get update && \
    apt-get install -y make && \
    rm -rf /var/lib/apt/lists/*

# copy all files
COPY . /home/slub/workspace

# install slub_docsa and dependencies
RUN cd /home/slub/workspace && make install-test

# run linting & tests
RUN cd /home/slub/workspace && make lint
RUN cd /home/slub/workspace && make coverage
