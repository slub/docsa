#!/bin/bash

coverage run -m --source=./src pytest
coverage report -m