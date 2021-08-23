# User Interfaces

This document discusses the scenarios and features of three main user interfaces that are intended to be developed
as part of this project.

Contents:
- [Intended User Interfaces](#intended-user-interfaces)
  - [Python Module API](#python-module-api)
  - [Linux Command Line Interface](#linux-command-line-interface)
  - [Http Query Interface via Annif](#http-query-interface-via-annif)
- [Future User Interfaces](#future-user-interfaces)


## Intended User Interfaces

### Python Module API

First and most importantly, the project should be accessible through a structured Python module interface.

Users
- Software Developers
- Machine Learning Developers

Scenarios
- Usage as traditional python library via `import slub_docsa`
  - Exexcute commands, e.g., import data, do predictions, evaluate a model
  - Add new data or import formats
  - Add or customize models and processing pipelines
  - Usage within a Jupyter Notebook for experimentation
  - Easy integration with other python libraries, e.g., `matplotlib`, `numpy`, `pandas`, etc.

Features
- API documentation of modules, classes and methods, e.g., via [sphinx](https://www.sphinx-doc.org/)
- Documented usage examples as tutorials or "getting started" description
- Mostly functional programming pattern that enables extensibility and customization
- Output results as simple data structures (lists, matrices) for easy post-processing with other python libraries

### Linux Command Line Interface

Secondly, in order to support easy evaluation and testing, a linux command line interface is developed.

Users
- Testers
- System Admins

Scenarios
- Testing and evaluation of various data sets, comparing models, etc.
- Setting up batch processes and cron jobs (e.g. regularly re-training of models)

Features (or CLI commands)
- Import new data (documents and classification schema)
- Train models, adjusting various parameters via CLI options
- Generate predictions for new documents via stdin/stdout (supporting Linux piping)
- Evaluate datasets and models, adjusting various parameters via CLI options

### Http Query Interface via Annif

Users
- Software Developers
- System Integrators

Scenarios
- Setting up a complex production system

Features
- Prediction for a single new document

Restrictions
- Annif only supports flat classes, i.e., a separate interface will be required to query for hierarchical information

## Future User Interfaces

Besides the intended user interfaces as described above, there are many other possible opportunities for integrations, e.g.:

- [Scikit-Learn](https://scikit-learn.org/) Pipeline Integration
- Integration with Machine Learning workflow tools, e.g., [Knime](https://www.knime.com/), [RapidMiner](https://rapidminer.com/)
- Language APIs, e.g., integrating with [R](https://www.r-project.org/)
- IBM [SPSS](https://www.ibm.com/analytics/spss-statistics-software) integration
