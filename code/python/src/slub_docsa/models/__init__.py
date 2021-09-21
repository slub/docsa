"""Classification models and similarity metrics.

## Dummy Models

### Oracle

### Random

### Stratified

## Scikit Models

### k-Nearest Neighbor

### Decision Tree

### Random Forest

### Multi-Layer Perceptron

### Logistic Regression

### Naive Bayes

### Support Vector Machine

## Annif Models

The following Annif models are supported by this implementation. Their respective name needs to be provided
as parameter `model_type` to `AnnifModel`. Performance metrics are calculated using an artificial random dataset
generated via `slub_docsa.data.artificial.hierarchical.generate_hierarchical_random_dataset` with X, Y, Z.

### Annif Tf-Idf

The `tfidf` model (see [description](https://github.com/NatLibFi/Annif/wiki/Backend%3A-TF-IDF) and
[implementation](https://github.com/NatLibFi/Annif/blob/master/annif/backend/tfidf.py)) uses the classic
term-frequency (tf) and inverse-document-frequency (idf) statistics for vectorization of both documents and subjects.

Requirements

- None

Training

- For each subject a corpus is generated. The corpus is constructed by combining all document texts that belong to
  a particular subject.
- All subject corpora are vectorized according to the tf-idf vectorization algorithm, see scikit-learn
  [TfidfVectorizer](
      https://scikit-learn.org/stable/modules/generated/sklearn.feature_extraction.text.TfidfVectorizer.html).

Prediction

- The document text is stemmed and vectorized according to the same tf-idf vectorization model.
- Then, the cosine similarity between the document and each subject is calculated (between 0 and 1).
- These similarity scores are returned as a probability score for each subject.

Limitations

- Since only similarities are calculated, this model is not able to learn any inter-subject relationships.

Performance

- `Todo` average top-3 f1-score, precision, recall

Literature

- `Todo`

### Annif Support Vector Classifier

The `svc` model (see [description](https://github.com/NatLibFi/Annif/wiki/Backend%3A-SVC) and
[implementation](https://github.com/NatLibFi/Annif/blob/master/annif/backend/svc.py)) uses the tf-idf vectorization
method and learns models using Support Vector Machines (SVM) with a linear kernel.

Requirements

- None

Training

- Documents are vectorized according to tf-idf statistics, see scikit-learn [TfidfVectorizer](
      https://scikit-learn.org/stable/modules/generated/sklearn.feature_extraction.text.TfidfVectorizer.html).
- A linear Support Vector Machine is trained using the scikit-learn library with default parameters, see [LinearSVC](
    https://scikit-learn.org/stable/modules/generated/sklearn.svm.LinearSVC.html)
- Each subject is represented by an independent support vector model in a one vs. rest strategy.

Prediction

- The new document is vectorized by the same tf-idf model.
- All SVC models for each subject are evaluated.
- Subject probabilities are calculated by considering the distance of the document to the decision hyperplane.

Limitations

- Support Vector Machines are binary classifier, and thus, will not be able to model and inter-subject relationships.
- Training does not support documents with multiple subject annotations. Instead, if there are multiple subject
  annotations, a "random" subject out of the available subject annotations is chosen as the target subject for that
  particular document. However, the randomness is not specifically implemented, but relies on the "randomness" of the
  `iter` strategy for the list of subject annotations.

Performance

- `Todo`

Literature

- `Todo`

### Annif Fasttext

The `fasttext` model (see [description](https://github.com/NatLibFi/Annif/wiki/Backend%3A-fastText) and
[implementation](https://github.com/NatLibFi/Annif/blob/master/annif/backend/fasttext.py)) uses Facebook's
[fastText](https://fasttext.cc/) library for training and predictions.

FastText is an alternative vectorization method that uses Artificial Neural Networks to learn word embeddings.

requires pip install

### Annif Omikuji

requires pip install

### Annif vw_multi

requires pip install

## Not Yet Supported Models

The following models are not yet supported or not yet tested.

### Annif Maui

### Annif Yake

`yake`

Requirements

- requires skos attribute, uses skos:"prefLabel", but no skos:"broader"

### Annif stwfsa

`stwfsa`

Requirements

- requires subjects as rdflib graph, supports using skos:"braoder"

### Annif mllm

Requirements

- requires subjects as rdflib graph, supports skos:"broader"

### Annif ensemble

### Annif pav

### Annif nn_ensemble
"""
