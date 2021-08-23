FROM python:3.7-slim

# setup python virtual environment
RUN python3 -m venv /home/slub/.venv
ENV PATH="/home/slub/.venv/bin:$PATH"
RUN python3 -m pip install --upgrade pip

# install development requirements
RUN pip install --no-cache-dir pylint flake8 flake8-docstrings pytest bandit pdoc3

# install python requirements
COPY code/python/requirements.txt /requirements.txt
RUN pip install --no-cache-dir -r /requirements.txt

# copy python src files
WORKDIR /home/slub/workspace
COPY code/python /home/slub/workspace/code/python

# set up python path
ENV PYTHONPATH=/home/slub/workspace/code/python/src

# create .cache dir required by pylint
RUN mkdir /home/slub/.cache

# run linting & tests
RUN cd code/python && sh ./lint.sh
RUN cd code/python && sh ./test.sh

# generate python documentation
RUN cd code/python && sh ./docs.sh
