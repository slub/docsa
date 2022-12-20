PYTHON=python3

.PHONY: deps
deps:
	apt-get update && apt-get install -y python3 python3-pip

.PHONY: install
install:
	# install and update pip, setuptools and wheel
	$(PYTHON) -m pip install --upgrade pip setuptools wheel
	# install slub_docsa package
	cd code/python/ && $(PYTHON) -m pip install .

.PHONY: install-test
install-test:
	# install and update pip, setuptools and wheel
	$(PYTHON) -m pip install --upgrade pip setuptools wheel
	# install slub_docsa package and test dependencies
	cd code/python/ && $(PYTHON) -m pip install .[test]

.PHONY: install-dev
install-dev:
	# install and update pip, setuptools and wheel
	$(PYTHON) -m pip install --upgrade pip setuptools wheel
	# install python all dependencies including developer dependencies
	cd code/python/ && $(PYTHON) -m pip install -e .[dev,test]

.PHONY: package
package:
	(cd code/python && bash package_build.sh)

.PHONY: serve
serve:
	slub_docsa serve

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

.PHONY: help
help:
	@echo ""
	@echo "  Targets"
	@echo ""
	@echo "    deps		Install python3 and pip via apt-get"
	@echo "    install		Install slub_docsa package and its dependencies via pip"
	@echo "    install-test	Install slub_docsa package and test dependencies via pip"
	@echo "    install-dev		Install slub_docsa package in edit mode (pip install -e .) for development"
	@echo "    package		Generates python source and wheel distributables"
	@echo "    lint		Checks python source code for lint problems"
	@echo "    test		Run unit tests"
	@echo "    coverage		Run unit tests and print coverage report"
	@echo "    docs		Generate python API documentation"
	@echo ""
