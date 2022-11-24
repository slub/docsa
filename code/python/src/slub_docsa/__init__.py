"""
This python package is a library for bibliographic document classification and similarity analysis.

It provides a selection of methods that support:

- pre-processing of bibliographic meta data and full-text documents,
- training of multi-label multi-class classification models,
- integrating and using hierarchical subject classifications (pruning methods, performance scores),
- similarity analysis and clustering.

Some important features include:

- a concise API for training and evaluating multi-label multi-class classification models, see `slub_docsa.common`
- support for many different classification approaches, see `slub_docsa.models`
- artificial hierarchical random datasets, see `slub_docsa.data.artificial`
- a performance score that considers hierarchical relations, see `slub_docsa.evaluation.score`

## Installation

This project requires [Python](https://www.python.org/) v3.8 or above and uses [pip](https://pypi.org/project/pip/)
for dependency management. Besides, this package uses [pyTorch](https://pytorch.org/) to train
[Artificial Neural Networks](https://en.wikipedia.org/wiki/Artificial_neural_network) via GPUs.
Make sure to install the latest Nvidia graphics drivers and check
[further requirements](https://pytorch.org/get-started/locally/#linux-prerequisites).

### Via Python Package Installer (not available yet)

Once published to PyPI (*not available yet*), install via:

- `python3 -m pip install slub_docsa`

### From Source

Download the source code by checking out the repository:

 - `git clone https://git.slub-dresden.de/lod/maschinelle-klassifizierung/docsa.git`

Use *make* to install python dependencies by executing the following commands:

- `make install` or `make install-test`
  (installs *slub_docsa* package and downloads all required runtime / test dependencies via *pip*)
- `make test`
  (runs tests to verify correct installation, requires test dependencies)
- `make docs`
  (generate API documentation, requires test dependencies)

### From Source using Ubuntu 20.04

Install essentials like *python3*, *pip* and *make*:

- `apt-get update`
   (update the Ubuntu package installer index)
- `apt-get install -y make python3 python3-pip`
   (install python3, pip and make)

Optionally, set up a python [virtual environment](https://docs.python.org/3/tutorial/venv.html):

- `apt-get install -y python3-venv`
- `python3 -m venv /path/to/venv`
- `source /path/to/venv/bin/activate`

Run *make* commands as provided above:

- `make install-test`
- `make test`

## First Steps

In order to get started, there are two possible approaches: the command line interface, or the Python API.

### Command Line Interface (CLI)

This library provides a single command `slub_docsa`, which supports to both train and evaluate classification and
clustering algorithms. A detailed description can be found in `slub_docsa.cli`.

### Python API

Besides the CLI, this library is follows a modular design such that new processing pipelines can be designed via the
Python API. A list of sub-modules can be found below. The most relevant classes and methods are:

- `slub_docsa.common.document.Document` - represents a document consisting of its title, abstract, fulltext
  and a list of authors
- `slub_docsa.common.subject.SubjectHierarchy` - models a hierarchy of subjects including labels for each subject
- `slub_docsa.common.dataset.Dataset` - combines documents and their subject annotations
- `slub_docsa.common.model.ClassificationModel` - defines how classification models are implemented and can be used
  for training and prediction, various model implementations can be found in `slub_docsa.models.classification`
- `slub_docsa.common.model.ClusteringModel` - defines how clustering models are implemented, various implements can be
  found in `slub_docsa.models.clustering`
- `slub_docsa.common.paths` - Storage configuration methods, which handle where various data is stored

### Example: Train a Model and Predict

Let's look at a simple example. The task is to train a classification model for the following documents:

```python
from slub_docsa.common.document import Document

documents = [
      Document(uri="uri://document1", title="This is a document title"),
      Document(uri="uri://document2", title="Document with interesting topics"),
      Document(uri="uri://document3", title="A boring title"),
]
```

In order to to be able to train a model, target subjects need to be known for said documents. Each subject is
referenced by its URI. Since this library supports multi-label annotations, each document may be associated with a
different number of subjects:

```python
subjects = [
      ["uri://subject1", "uri://subject2"],                   # document 1
      ["uri://subject1", "uri://subject3", "uri://subject4"], # document 2
      ["uri://subject2"],                                     # document 3
]
```

Not that both lists need to follow the same order, meaning document `documents[i]` is annotated with subjects at
`subjects[i]`. Together, they form the dataset that is used for training and evaluation:

```python
from slub_docsa.common.dataset import SimpleDataset

dataset = SimpleDataset(documents=documents, subjects=subjects)
```

Many machine learning algorithms operate based on a vector representation instead of text. In this case, a
vectorization method needs to be selected. Various implementations can be found in
`slub_docsa.data.preprocess.vectorizer`.
In this example, we choose the `slub_docsa.data.preprocess.vectorizer.TfidfVectorizer`, which is based on the scikit
implementation of the same name, see [sklearn.feature_extraction.text.TfidfVectorizer](
https://scikit-learn.org/stable/modules/generated/sklearn.feature_extraction.text.TfidfVectorizer.html).

```python
from slub_docsa.data.preprocess.vectorizer import TfidfVectorizer

vectorizer = TfidfVectorizer()
```

As a classification model we can choose from a number of existing implementations, see
`slub_docsa.models.classification`. In this example, we begin with a basic approach called k-nearest neighbor
classifier, which again is provided by the scikit library, see [sklearn.neighbors.KNeighborsClassifier](
https://scikit-learn.org/stable/modules/generated/sklearn.neighbors.KNeighborsClassifier.html).
In order to interface with scikit, the wrapper `slub_docsa.models.classification.scikit.ScikitClassifier` can be used:

```python
from slub_docsa.models.classification.scikit import ScikitClassifier
from sklearn.neighbors import KNeighborsClassifier

model = ScikitClassifier(
    predictor=KNeighborsClassifier(n_neighbors=2),
    vectorizer=vectorizer
)
```

Before training the model, the target subject annotations need to be transformed to a matrix representation, which
encodes which document is annotated with which subject, often called incidence matrix.

```python
from slub_docsa.evaluation.incidence import subject_incidence_matrix_from_targets
from slub_docsa.evaluation.incidence import unique_subject_order

subject_order = unique_subject_order(dataset.subjects)
incidence_matrix = subject_incidence_matrix_from_targets(
    dataset.subjects,
    subject_order
)

print(subject_order)
print(incidence_matrix)
```

The result will be a fixed ordering of subjects and a matrix that encodes which document (rows) is annotated by which
subject (columns):

```python
['uri://subject1', 'uri://subject3', 'uri://subject2', 'uri://subject4']
[[1. 0. 1. 0.]
 [1. 1. 0. 1.]
 [0. 0. 1. 0.]]
```

Then, the model can be trained:

```python
model.fit(dataset.documents, incidence_matrix)
```

In order to predict subject for new documents, we can provide a list of yet unknown documents.

```python
new_documents = [
    Document(uri="uri://new_document1", title="Title of the new document"),
    Document(uri="uri://new_document2", title="Boring subject"),
]

predicted_probabilities = model.predict_proba(new_documents)
print(predicted_probabilities)
```

The result will be a probability matrix, which encodes which of the new documents is associated with which subject and
to what degree (as a value between 0 and 1):

```python
[[0.5 0.  1.  0. ]
 [0.5 0.5 0.5 0.5]]
```

In order to calculate final prediction decision, a decision strategy needs to be applied to the probability scores,
which binarizes the matrix to an incidence matrix. Several strategies, e.g., based on a threshold or based on choosing
the top-k subjects are implemented in `slub_docsa.evaluation.incidence`. In this example, we apply the top-2 decision
strategy, which chooses two subjects with highest probability as the output:

```python
from slub_docsa.evaluation.incidence import top_k_incidence_decision

incidence_decision_function = top_k_incidence_decision(k=2)
predicted_incidence = incidence_decision_function(predicted_probabilities)
print(predicted_incidence)
```

The result will be a binary incidence matrix:

```python
[[1. 0. 1. 0.]
 [0. 0. 1. 1.]]
```

Note that the decision for the second document will be random, since all subjects have equal probability according to
the k-nearest neighor model.

In order to retrieve the actual subjects represented by this incidence matrix, we can apply the following reverse
function:

```python
from slub_docsa.evaluation.incidence import subject_targets_from_incidence_matrix

predicted_subjects = subject_targets_from_incidence_matrix(predicted_incidence, subject_order)
print(predicted_subjects)
```

The output will be:

```python
[['uri://subject1', 'uri://subject2'], ['uri://subject2', 'uri://subject4']]
```

Suppose the true subjects for both documents would be known, we can calculate several performance scores given both the
predicted incidence and true incidence. The most common performance scores are precision, recall and f1-score. Again,
this library provides an interface `slub_docsa.evaluation.score.scikit_incidence_metric` to the scikit-learn library,
such that scores can be easily calculated:

```python
from slub_docsa.evaluation.score import scikit_incidence_metric
from sklearn.metrics import f1_score

true_subjects = [
    ["uri://subject1", "uri://subject2"],
    ["uri://subject4"]
]

true_incidence = subject_incidence_matrix_from_targets(true_subjects, subject_order)

score = scikit_incidence_metric(
    incidence_decision_function, f1_score, average="micro"
)(
    true_incidence, predicted_probabilities
)
print("f1 score is", score)
```

The output will be:

```python
f1 score is 0.8571428571428571
```

### Example: Compare Models and Plot

The following example describes how a single dataset can be evaluate by comparing multiple different models based on
multiple scores via cross-validation. Similar to before, we first have to define the dataset that is being used.
In this example, let's use an aritificially generated hierarchical dataset, see
`slub_docsa.data.artificial.hierarchical`.

The dataset is generated based on a selection of tokens (or words). In order to extract common english words, we use
DBpedia as a resource. We download and iterate over 1000 abstracts from DBpedia to extract common english words via
`slub_docsa.data.artificial.tokens.token_probabilities_from_dbpedia`.

```python
from slub_docsa.data.artificial.tokens import token_probabilities_from_dbpedia

token_probabilities = token_probabilities_from_dbpedia("en", n_docs=1000)

print(len(token_probabilities))
```

The result will be 10.079 different common english tokens (or words).

```python
10079
```

Each token is further characterized by its probability of occurrence within DBpedia. For example, the following 5
tokens are extracted with their respective occurrence probability:

```python
for t in list(token_probabilities.keys())[:5]:
    print(t, ":", token_probabilities[t])
```

The output will be:

```python
animalia : 1.575721286418858e-05
is : 0.023509761593369365
an : 0.00877676756535304
illustrated : 3.151442572837716e-05
book : 0.00039393032160471456
```

Based on these simple token statistics, we can generate 1000 random artificial documents and 10 random artificial
hierarchical subjects:

```python
from slub_docsa.data.artificial.hierarchical import generate_hierarchical_random_dataset_from_token_probabilities

dataset, subject_hierarchy = generate_hierarchical_random_dataset_from_token_probabilities(
    token_probabilities, n_documents=1000, n_subjects=10
)

print(dataset.documents[0])
```

Each document will consist of a sequence of random english tokens (or words):

```python
<Document
  title="practical of molecule with greek along particular include second known and power alexandre was to ag"
  abstract="None"
  fulltext="None"
>
```

Also, each document is assigned to one or multiple artificial subjects structured in a simple hierarchy:

```python
from slub_docsa.common.subject import print_subject_hierarchy

print_subject_hierarchy("en", subject_hierarchy)
```

The output will be:

```python
uri://random/subject/1
    uri://random/subject/7
    uri://random/subject/6
    uri://random/subject/5
    uri://random/subject/4
uri://random/subject/2
    uri://random/subject/10
    uri://random/subject/9
    uri://random/subject/8
uri://random/subject/3
```

In order to be able to apply cross-validation in a meaningful way, a minimum number of examples for each subject is
desired. Since documents and their subject annotations are generated randomly, it is possible that one or more subjects
do not have sufficient examples assigned to them. In the following, we will check that each subject consists of at
least 10 examples. Subjects that consist of less than 10 examples will be pruned.

```python
from slub_docsa.data.preprocess.dataset import filter_subjects_with_insufficient_samples
from slub_docsa.data.preprocess.subject import prune_subject_targets_to_minimum_samples

min_samples = 10

dataset.subjects = prune_subject_targets_to_minimum_samples(
    min_samples, dataset.subjects, subject_hierarchy
)
dataset = filter_subjects_with_insufficient_samples(
    dataset, min_samples
)
```

In order to verify whether some subjects have been pruned, we can calculate the unique subject list:

```python
from slub_docsa.evaluation.incidence import unique_subject_order

subject_order = unique_subject_order(dataset.subjects)
print(len(subject_order))
```

Depending on the random generation process, the result may vary. In this case, one subject was pruned.

```python
9
```

Next, we need to define multiple classification models that will be evaluated:

```python
from slub_docsa.models.classification.dummy import NihilisticModel, OracleModel
from slub_docsa.models.classification.scikit import ScikitClassifier
from slub_docsa.data.preprocess.vectorizer import TfidfStemmingVectorizer
from sklearn.neighbors import KNeighborsClassifier
from sklearn.metrics import f1_score

models = [
    OracleModel(),
    NihilisticModel(),
    ScikitClassifier(
        predictor=KNeighborsClassifier(n_neighbors=1),
        vectorizer=TfidfStemmingVectorizer("en", max_features=2000),
    )
]
```

Since the predictive performance can be evaluated in various ways, we also need to define multiple score functions:

```python
from slub_docsa.evaluation.score import scikit_metric_for_best_threshold_based_on_f1score
from slub_docsa.evaluation.score import cesa_bianchi_h_loss, scikit_incidence_metric
from slub_docsa.evaluation.incidence import positive_top_k_incidence_decision

scores = [
    # f1 score for best threshold
    scikit_metric_for_best_threshold_based_on_f1score(
        f1_score, average="micro", zero_division=0
    ),
    # f1 score for top-3 selection
    scikit_incidence_metric(
        positive_top_k_incidence_decision(3),
        f1_score,
        average="micro",
        zero_division=0
    ),
    # hierarchical loss
    scikit_metric_for_best_threshold_based_on_f1score(
        cesa_bianchi_h_loss(subject_hierarchy, subject_order, log_factor=1000),
    )
]
```

Finally, we can apply cross-validation to our dataset. In order to get additional information during the evaluation,
we can optionally set up the python logging library:

```python
import logging
logging.basicConfig(level=logging.INFO)
```

Next, we will trigger an evaluation pipeline, which will train each model multiple times for each cross-validation
split, and calculate each score function by comparing the model predictions with the test dataset.

```python
from slub_docsa.evaluation.pipeline import score_classification_models_for_dataset
from slub_docsa.evaluation.split import scikit_kfold_splitter

n_splits = 10
split_function = scikit_kfold_splitter(n_splits)

score_matrix, _ = score_classification_models_for_dataset(
    n_splits,
    dataset,
    subject_order,
    models,
    split_function,
    scores,
    [],
)
```

The console output will contain a number of status messages reporting the evaluation progress:

```python
INFO:slub_docsa.evaluation.pipeline:prepare 1-th cross validation split
INFO:slub_docsa.evaluation.pipeline:evaluate 1-th cross validation split with 900 training and 100 test samples
INFO:slub_docsa.evaluation.pipeline:evaluate model <OracleModel> for 1-th split
INFO:slub_docsa.evaluation.pipeline:do training
INFO:slub_docsa.evaluation.pipeline:do prediction
INFO:slub_docsa.evaluation.pipeline:do global scoring
INFO:slub_docsa.evaluation.pipeline:do per-subject scoring
INFO:slub_docsa.evaluation.pipeline:overall scores are: [1. 1. 0.]
...
```

In some cases, the cross-validation splits will result in training and test sets that are not well balanced with
respect to one or more subjects. In these cases warning messages might be shown:

```python
WARNING:slub_docsa.evaluation.condition:subject 'uri://random/subject/3' is outside
    target split ratio interval of (0.05, 0.2) with 18 training and 5 test samples
```

The resulting score matrix contains each score value for each model and cross-validation split. In order to visualize
the score matrix, it can be plotted:

```python
from slub_docsa.evaluation.plotting import score_matrix_box_plot
from slub_docsa.evaluation.plotting import write_multiple_figure_formats

figure = score_matrix_box_plot(
    score_matrix,
    model_names=["oracle", "nihilistic", "knn k=1"],
    score_names=["t=best f1_score", "top-3 f1_score", "h-loss"],
    score_ranges=[(0, 1), (0, 1), (0, None)]
)

write_multiple_figure_formats(figure, filepath="example_score_matrix")
```

The result will be three files that illustrate the performance of each model in a box plot:

- `example_score_matrix.html`
- `example_score_matrix.jpg`
- `example_score_matrix.pdf`

"""

import os
import subprocess  # nosec


def _git_commit_hash():
    """Return the shortened git commit hash of this repository or an empty string if not available."""
    try:
        return "-" + subprocess.check_output(  # nosec
            ["git", "rev-parse", "--short", "HEAD"],
            cwd=os.path.dirname(os.path.abspath(__file__))).strip().decode()
    except Exception:  # pylint: disable=broad-except
        return ""


__version__ = "0.1.0.dev1" + _git_commit_hash()
