"""Base class describing a model."""


from typing import Sequence, Iterable

from slub_docsa.common.document import Document


class Model:
    """Represents a model similar to a scikit-learn estimator and predictor interface.

    However, the input of both fit, predict and predict_proba methods are a collection of `Document` instances,
    instead of raw vectorized feaures.
    """

    def fit(self, train_documents: Sequence[Document], train_targets: Sequence[Iterable[str]]):
        """Train a model to fit the training data.

        Parameters
        ----------
        train_documents: Sequence[Document]
            The sequence of documents that is used for training a model.
        train_targets: Sequence[Iterable[str]]
            The sequence of subject URIs for each document of `train_documents`.

        Returns
        -------
        Model
            self
        """
        raise NotImplementedError()

    def predict(self, test_documents: Sequence[Document]) -> Sequence[Iterable[str]]:
        """Return hard predictions without probabilities as incidence matrix.

        Parameters
        ----------
        test_documents: Sequence[Document]
            The test sequence of documents that are supposed to be evaluated.

        Returns
        -------
        Sequence[Iterable[str]]
            The list of subject URIs predicted for each test document.
        """
        raise NotImplementedError()
