PYTHON=python3

.PHONY: deps
deps:
	$(PYTHON) -m pip install --no-cache-dir -r code/python/requirements.txt

.PHONY: deps-test
deps-test:
	$(PYTHON) -m pip install --no-cache-dir pylint flake8 flake8-docstrings pytest pytest-watch coverage bandit pdoc3

.PHONY: lint
lint:
	(cd code/python && bash lint.sh)

.PHONY: test
test:
	(cd code/python && bash test_run.sh)

.PHONY: coverage
coverage:
	(cd code/python && bash test_coverage.sh)

.PHONY: docs
docs:
	(cd code/python && bash docs.sh)