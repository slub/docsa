# Workflow Design

## Linux Command Line Interface

### Import Data

Datasets and variants of datasets (e.g. subsets, versions) are managed via an identifier string `<name>`.

- `slub_docsa dataset add <name> <format> [options]`

  Adds a new data set with name `<name>` using some `<format>` and optional `[options]` depending on the format.
  - `<name>` is a simple string describing the name of the data set, e.g., `qucosa`
  - `<format>` is a string referencing a python class subclassing the dataset interface, e.g., `tsv` or `qucosa`
  - `[options]` are various arguments depending on the format, e.g., `--document-tsv <path>` and `--classes-tsv <path>`.

- `slub_docsa dataset list`

  Prints a list of currently available datasets.
  ```
  DATASET               | FORMAT | OPTIONS
  --------------------- | ------ | ------------------------------------
  qucosa_rvk_all        | qucosa | --subject-type=rvk
  qucosa_rvk_notation_A | qucosa | --subject-type=rvk --rvk_notation=A*
  qucosa_rvk_depth_2    | qucosa | --subject-type=rvk --subject-depth=2
  ```

- `slub_docas dataset formats`

  Prints a list of currently availabel dataset formats.
  ```
  FORMAT | DESCRIPTION
  ------ | ---------------------------------------------------
  tsv    | Tab Separates Values files, see docs/formats/tsv.md
  qucosa | Qucosa Json files, see docs/formats/qucosa.md
  ```

- `slub_docsa dataset remove <name>`

  Removes a dataset with name `<name>`. Will remove all cached data structures, etc.

### Classification Training, Prediciton and Evaluation

- `slub_docsa classifier add <name> <type> <dataset> [options]`
- `slub_docsa classifier train <name> [options]`
- `slub_docsa classifier predict <name> [options]`
- `slub_docsa classifier evaluate <name> [options]`
- `slub_docsa classifier types`
- `slub_docsa classifier list`
- `slub_docsa classifier remove <name>`

### Similarity Analysis and Clustering

- `slub_docsa similarity add <name> <type> <dataset> [options]`
- `slub_docsa similarity train <name> [options]`
- `slub_docsa similarity predict <name> [options]`
- `slub_docsa similarity cluster <name> [options]`
- `slub_docsa similarity types`
- `slub_docsa similarity list`
- `slub_docsa similarity remove <name>`

### Missing Features

- Transfer Learning between datasets
- Data statistics (class distribution, feature distribution, feature correlation, etc.)