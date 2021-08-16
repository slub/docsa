FROM python:3.6-slim

# setup python virtual environment
RUN python3 -m venv /home/slub/.venv
ENV PATH="/home/slub/.venv/bin:$PATH"
RUN python3 -m pip install --upgrade pip

# install development requirements
RUN pip install --no-cache-dir pylint flake8 pytest

# install python requirements
COPY code/python/requirements.txt /requirements.txt
RUN pip install --no-cache-dir -r /requirements.txt

# copy python src files
WORKDIR /home/slub/workspace
COPY code/python /home/slub/workspace/code/python

# set up python path
ENV PYTHONPATH=/home/slub/workspace/code/python/src

# run linting & tests
RUN cd code/python && sh ./lint.sh
RUN cd code/python && sh ./test.sh
