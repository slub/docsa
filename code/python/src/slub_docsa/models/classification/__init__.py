"""Classification models.

Classification models implement the `slub_docsa.common.model.ClassificationModel` interface and provide various
algorithms for multi-class multi-label classification.

## Dummy Models

Dummy models do not actually perform any intelligent learning process, but are used as a baseline and for testing
purposes.

### Oracle

The `slub_docsa.models.classification.dummy.OracleModel` is a model implementation that cheats by getting access to the
target subjects of the test dataset. Instead of learning a model it simply returns the true subject annotations.

As a result, this model should always outperform any other classification model. This can be useful to determine
the maximal achievable score in some scenarios where the best score is not known in advance.

For example, when chosing subjects based on the
`slub_docsa.evaluation.classification.incidence.PositiveTopkIncidenceDecision` and there are documents that are
annotated with more then `k` subjects, even the `slub_docsa.models.classification.dummy.OracleModel` will not be able
to achieve the maximum f1 score of `1.0`, since some subjects are not predicted even if their test subject
probabilities are correctly predicted as `1.0`.

### Nihilistic

The `slub_docsa.models.classification.dummy.NihilisticModel` simply returns a subject probability of `0.0` for all
predictions.

### Random

The `slub_docsa.models.classification.dummy.RandomModel` returns uniform random subject probabilities between `0.0` and
`1.0`.

This means, for example, using the `slub_docsa.evaluation.classification.incidence.ThresholdIncidenceDecision` with
`threshold=0.5`, it will predict on average half of all available subjects.

Random predictions can be useful as a minimum baseline and to determine minimum score values. Ideally, non-dummy models
should outperform the random model.

## Scikit Models

Many classic classification algorithms have already been implemented as part of the [scikit-learn](
https://scikit-learn.org/) library. The `slub_docsa.models.classification.scikit.ScikitClassifier` model provides an
interface to these classic algorithms.

Unfortunately, not all classifier can be used since not all of them support multi-class multi-label classifications.
For example, the [`LogisticRegression`](
https://scikit-learn.org/stable/modules/generated/sklearn.linear_model.LogisticRegression.html) classifier does not
support multi-label predictions. However, it can be extended by the [`MultiOutputClassifier`](
https://scikit-learn.org/stable/modules/generated/sklearn.multioutput.MultiOutputClassifier.html) wrapper, which then
trains multiple models using the one-vs-rest strategy.

Similarily, not all classifiers support predicting class probabilities. For example, the [`LinearSVC`](
https://scikit-learn.org/stable/modules/generated/sklearn.svm.LinearSVC.html) does not implement the `predict_proba`
method, and thus, can not be used to determine class probabilities. However, such classifier can be extended with the
[`CalibratedClassifierCV`](
https://scikit-learn.org/stable/modules/generated/sklearn.calibration.CalibratedClassifierCV.html) wrapper, which
estimates class probabilities via cross validation.

An overview over supported multi-class multi-label classifiers can be found [here](
https://scikit-learn.org/stable/modules/multiclass.html).

### Dummy Classifier

Similar to the dummy models provided above, [scikit-learn](https://scikit-learn.org/) also provides various dummy
strategies with the [`DummyClassifier`](
https://scikit-learn.org/stable/modules/generated/sklearn.dummy.DummyClassifier.html). Again, these models can be
useful as a baseline for comparison, or for testing purposes.

## Annif Models

The following Annif models are supported by this implementation. Their respective name needs to be provided
as parameter `model_type` to `AnnifModel`.

### Annif Tf-Idf

The `tfidf` model (see [description](https://github.com/NatLibFi/Annif/wiki/Backend%3A-TF-IDF) and
[implementation](https://github.com/NatLibFi/Annif/blob/master/annif/backend/tfidf.py)) uses the classic
term-frequency (tf) and inverse-document-frequency (idf) statistics for vectorization of both documents and subjects.

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

### Annif Support Vector Classifier

The `svc` model (see [description](https://github.com/NatLibFi/Annif/wiki/Backend%3A-SVC) and
[implementation](https://github.com/NatLibFi/Annif/blob/master/annif/backend/svc.py)) uses the tf-idf vectorization
method and learns models using Support Vector Machines (SVM) with a linear kernel.

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

### Annif Fasttext

The `fasttext` model (see [description](https://github.com/NatLibFi/Annif/wiki/Backend%3A-fastText) and
[implementation](https://github.com/NatLibFi/Annif/blob/master/annif/backend/fasttext.py)) uses Facebook's
[fastText](https://fasttext.cc/) library for training and predictions.

FastText is an alternative vectorization method that uses Artificial Neural Networks to learn word embeddings.

## Not Yet Supported Models

The following models are not yet supported or not yet tested.

- Annif Maui
- Annif stwfsa
- Annif ensemble
- Annif pav
- Annif nn_ensemble
"""
