#!/bin/bash

(cd ./src && coverage run -m --source=. pytest ../tests)
(cd ./src && coverage report -m)