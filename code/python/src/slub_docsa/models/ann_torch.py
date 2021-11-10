"""Torch Models for Classification."""

# pylint: disable=fixme, invalid-name, no-member, too-many-locals, too-many-statements

import logging
import time

from typing import Any, Optional, Sequence, cast

import numpy as np

import torch
import scipy

from sklearn.metrics import f1_score

from torch.nn.modules.activation import ReLU, Tanh
from torch.utils.data import TensorDataset, DataLoader
from torch.nn import Sequential, Linear, Dropout, BCEWithLogitsLoss
from torch.optim import Adam
from torch.optim.lr_scheduler import ExponentialLR

from slub_docsa.common.model import Model
from slub_docsa.common.document import Document
from slub_docsa.data.preprocess.document import document_as_concatenated_string
from slub_docsa.data.preprocess.vectorizer import AbstractVectorizer
from slub_docsa.evaluation.score import scikit_metric_for_best_threshold_based_on_f1score

logger = logging.getLogger(__name__)


class AbstractTorchModel(Model):
    """A abstract torch model.

    Implement the `get_model` method to provide your custom network model.
    """

    def __init__(
        self,
        vectorizer: AbstractVectorizer,
        epochs: int = 50,
        batch_size: int = 32,
        lr: float = 0.001
    ):
        """Initialize model.

        Parameters
        ----------
        vectorizer: AbstractVectorizer
            the vectorizer used to transform documents to features vectors
        epochs: int
            the number of epochs used for training
        batch_size: int
            the number examples used to calculate a gradient as a single batch
        lr: float
            the learning rate
        """
        self.vectorizer = vectorizer
        self.epochs = epochs
        self.model = None
        self.batch_size = batch_size
        self.lr = lr
        self.device = "cuda" if torch.cuda.is_available() else "cpu"

    def get_model(self, n_inputs, n_outputs) -> torch.nn.Module:
        """Return a torch network that will be trained and evaluated."""
        raise NotImplementedError()

    def fit(
        self,
        train_documents: Sequence[Document],
        train_targets: np.ndarray,
        validation_documents: Optional[Sequence[Document]] = None,
        validation_targets: Optional[np.ndarray] = None,
    ):
        """Train the fully connected network for all training documents."""
        # transform documents to feature vectors
        logger.info("train fully connected network with %d examples", len(train_documents))
        self.vectorizer.fit(document_as_concatenated_string(d) for d in train_documents)
        train_features = list(self.vectorizer.transform(document_as_concatenated_string(d) for d in train_documents))
        train_features = np.array(train_features)  # make sure feature vectors is a full numpy array

        # extract and prepare validation features
        validation_features = None
        if validation_documents is not None:
            validation_features = list(
                self.vectorizer.transform(document_as_concatenated_string(d) for d in validation_documents)
            )
            validation_features = np.array(validation_features)

        # convert to tensors
        train_features_tensor = torch.from_numpy(train_features).float()
        train_targets_tensor = torch.from_numpy(train_targets).float()

        validation_features_tensor = None
        validation_targets_tensor = None
        if validation_features is not None:
            validation_features_tensor = torch.from_numpy(validation_features).float()
            validation_targets_tensor = torch.from_numpy(validation_targets).float()

        # setup torch datatsets, such that data can be loaded in batches
        train_dataset = TensorDataset(train_features_tensor, train_targets_tensor)
        train_dataloader = DataLoader(train_dataset, batch_size=self.batch_size, shuffle=True)

        validation_dataset = None
        validation_dataloader = None
        if validation_features_tensor is not None and validation_targets_tensor is not None:
            validation_dataset = TensorDataset(validation_features_tensor, validation_targets_tensor)
            validation_dataloader = DataLoader(validation_dataset, batch_size=self.batch_size, shuffle=False)

        # initialize network model
        logger.info("initialize torch model on device '%s'", self.device)
        self.model = self.get_model(int(train_features.shape[1]), int(train_targets.shape[1]))
        self.model.to(self.device)
        self.model.train()

        # define loss and optimizer
        criterion = BCEWithLogitsLoss()
        optimizer = Adam(self.model.parameters(), lr=self.lr)
        scheduler = ExponentialLR(optimizer, gamma=0.99)

        last_validation_time = time.time()

        # iterate over epochs and batches
        for epoch in range(self.epochs):

            # do training
            loss: Any = None
            batch = 0
            epoch_train_loss = 0
            for batch, (X, y) in enumerate(train_dataloader):
                # send features and targets to device
                X, y = X.to(self.device), y.to(self.device)

                # calculate loss
                output = self.model(X)
                loss = criterion(output, y)

                # do backpropagation
                optimizer.zero_grad()
                loss.backward()
                optimizer.step()

                epoch_train_loss += loss.item()

            epoch_train_loss = epoch_train_loss / (batch + 1)

            # calculate test error on validation data
            epoch_validation_loss = None
            epoch_validation_f1_score = None
            if validation_dataloader is not None and validation_targets is not None \
                    and time.time() - last_validation_time > 0.0:
                self.model.eval()
                output_arrays = []
                with torch.no_grad():
                    epoch_validation_loss = 0
                    batch = 0
                    for batch, (X, y) in enumerate(validation_dataloader):
                        X, y = X.to(self.device), y.to(self.device)
                        output = self.model(X)
                        loss = criterion(output, y)
                        epoch_validation_loss += loss.item()

                        output_arrays.append(output.cpu().detach().numpy())

                    epoch_validation_loss = epoch_validation_loss / (batch + 1)

                validation_probabilities = cast(Any, scipy).special.expit(np.vstack(output_arrays))
                logger.debug(
                    "validation probabilities shape %s vs validation targets shape %s",
                    validation_probabilities.shape,
                    validation_targets.shape
                )
                epoch_validation_f1_score = scikit_metric_for_best_threshold_based_on_f1score(
                    f1_score, average="micro", zero_division=0
                )(validation_targets, validation_probabilities)

                last_validation_time = time.time()
                self.model.train()

            logger.debug(
                "trained model for epoch %d with %d batches, train loss of %s, test loss of %s, test f1_score of %s",
                epoch, batch, epoch_train_loss, epoch_validation_loss, epoch_validation_f1_score
            )

            scheduler.step()
            logger.debug("learning rate is %s", optimizer.param_groups[0]["lr"])

    def predict_proba(self, test_documents: Sequence[Document]) -> np.ndarray:
        """Predict class probabilities for all test documents."""
        if not self.model:
            raise ValueError("no model trained yet")

        # transform documents to feature vectors
        features = list(self.vectorizer.transform(document_as_concatenated_string(d) for d in test_documents))

        # make sure feature vectors is a full numpy array
        # TODO: convert sparse matrices to sparse tensors
        # if isinstance(features, csr_matrix):
        #    features = features.toarray()
        features = np.array(features)

        # convert to tensors
        features_tensor = torch.from_numpy(features).float()

        # setup torch datatsets
        torch_dataset = TensorDataset(features_tensor)
        dataloader = DataLoader(torch_dataset, batch_size=self.batch_size)

        # iterate over batches of all examples
        arrays = []
        self.model.eval()
        with torch.no_grad():
            for X in dataloader:
                # send each examples to device
                Xs = [x.to(self.device) for x in X]
                # evaluate model for each test example
                outputs = self.model(*Xs)
                # retrieve outputs and collected them as numpy arrays
                array = outputs.cpu().detach().numpy()
                arrays.append(array)

        # reverse logits and return results
        predictions = cast(np.ndarray, cast(Any, scipy).special.expit(np.vstack(arrays)))
        logger.debug("predictions shape is %s", predictions.shape)
        return predictions

    def __str__(self):
        """Return representative string for model."""
        return f"<{self.__class__.__name__} vectorizer={str(self.vectorizer)} " + \
            f"epochs={self.epochs} batch_size={self.batch_size} lr={self.lr}>"


class TorchBertSequenceClassificationHeadModel(AbstractTorchModel):
    """A torch model that is follows the classification head of a Bert Sequence Classification network.

    See HuggingFace: https://huggingface.co/transformers/_modules/transformers/modeling_bert.html
    """

    def get_model(self, n_inputs, n_outputs):
        """Return the sequence classification head model."""
        return Sequential(
            # BertPooler
            Linear(n_inputs, n_inputs),
            Tanh(),
            Dropout(p=0.1),
            # Classifier
            Linear(n_inputs, n_outputs),
        )


class TorchSingleLayerDenseModel(AbstractTorchModel):
    """A simple torch model consisting of one hidden layer of 1024 neurons with ReLU activations."""

    def get_model(self, n_inputs, n_outputs):
        """Return the linear network."""
        return Sequential(
            Linear(n_inputs, 1024),
            ReLU(),
            Linear(1024, n_outputs),
        )
