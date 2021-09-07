FROM python:3.7-slim

# install make
RUN apt-get update && \
    apt-get install -y make && \
    rm -rf /var/lib/apt/lists/*

# copy make file and requirements to install deps
WORKDIR /home/slub/workspace
COPY code/python/requirements.txt /home/slub/workspace/code/python/requirements.txt
COPY Makefile /home/slub/workspace/Makefile

# install python requirements
RUN make deps
RUN make deps-test

# copy all python src files
COPY code/python /home/slub/workspace/code/python

# run linting & tests
RUN make lint
RUN make coverage

# generate python documentation
RUN make docs
