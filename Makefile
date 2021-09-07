PYTHON=python3

deps:
	$(PYTHON) -m pip install --no-cache-dir -r code/python/requirements.txt

deps-test:
	$(PYTHON) -m pip install --no-cache-dir pylint flake8 flake8-docstrings pytest bandit pdoc3

.PHONY: test
test:
	(cd code/python && bash lint.sh)
	(cd code/python && bash test_run.sh)

coverage:
	(cd code/python && bash test_coverage.sh)

.PHONY: docs
docs:
	(cd code/python && bash docs.sh)