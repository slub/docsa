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

This project requires [Python](https://www.python.org/) v3.6 or above and uses [pip](https://pypi.org/project/pip/)
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

### From Source using Ubuntu 18.04

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
- `slub_docsa.common.subject.SubjectNode` - models a hierarchical subject consisting of its URI and label and parent
  subject
- `slub_docsa.common.dataset.Dataset` - combines documents and their subject annotations
- `slub_docsa.common.model.ClassificationModel` - defines how classification models are implemented and can be used
  for training and prediction, various model implementations can be found in `slub_docsa.models.classification`
- `slub_docsa.common.model.ClusteringModel` - defines how clustering models are implemented, various implements can be
  found in `slub_docsa.models.clustering`
- `slub_docsa.common.paths` - Storage configuration methods, which handle where various data is stored

### An Example

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

```
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

```
['uri://subject1', 'uri://subject3', 'uri://subject2', 'uri://subject4']
[[1. 0. 1. 0.]
 [1. 1. 0. 1.]
 [0. 0. 1. 0.]]
```

Then, the model can be trained:

```
model.fit(dataset.documents, incidence_matrix)
```

In order to predict subject for new documents, we can provide a list of yet unknown documents.

```
new_documents = [
    Document(uri="uri://new_document1", title="Title of the new document"),
    Document(uri="uri://new_document2", title="Boring subject"),
]

predicted_probabilities = model.predict_proba(new_documents)
print(predicted_probabilities)
```

The result will be a probability matrix, which encodes which of the new documents is associated with which subject and
to what degree (as a value between 0 and 1):

```
[[0.5 0.  1.  0. ]
 [0.5 0.5 0.5 0.5]]
```

In order to calculate final prediction decision, a decision strategy needs to be applied to the probability scores,
which binarizes the matrix to an incidence matrix. Several strategies, e.g., based on a threshold or based on choosing
the top-k subjects are implemented in `slub_docsa.evaluation.incidence`. In this example, we apply the top-2 decision
strategy, which chooses two subjects with highest probability as the output:

```
from slub_docsa.evaluation.incidence import top_k_incidence_decision

incidence_decision_function = top_k_incidence_decision(k=2)
predicted_incidence = incidence_decision_function(predicted_probabilities)
print(predicted_incidence)
```

The result will be a binary incidence matrix:

```
[[1. 0. 1. 0.]
 [0. 0. 1. 1.]]
```

Note that the decision for the second document will be random, since all subjects have equal probability according to
the k-nearest neighor model.

In order to retrieve the actual subjects represented by this incidence matrix, we can apply the following reverse
function:

```
from slub_docsa.evaluation.incidence import subject_targets_from_incidence_matrix

predicted_subjects = subject_targets_from_incidence_matrix(predicted_incidence, subject_order)
print(predicted_subjects)
```

The output will be:

```
[['uri://subject1', 'uri://subject2'], ['uri://subject2', 'uri://subject4']]
```

Suppose the true subjects for both documents would be known, we can calculate several performance scores given both the
predicted incidence and true incidence. The most common performance scores are precision, recall and f1-score. Again,
this library provides an interface `slub_docsa.evaluation.score.scikit_incidence_metric` to the scikit-learn library,
such that scores can be easily calculated:

```
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

```
f1 score is 0.8571428571428571
```

"""
