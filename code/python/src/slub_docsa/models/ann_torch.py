"""Torch Models for Classification."""

# pylint: disable=fixme, invalid-name, no-member, too-many-locals

import logging

from typing import Any, Sequence, cast

import numpy as np
import torch
import scipy

from torch.nn.modules.activation import ReLU, Tanh
from torch.utils.data import TensorDataset, DataLoader
from torch.nn import Sequential, Linear, Dropout, BCEWithLogitsLoss
from torch.optim import Adam

from slub_docsa.common.model import Model
from slub_docsa.common.document import Document
from slub_docsa.data.preprocess.document import document_as_concatenated_string
from slub_docsa.data.preprocess.vectorizer import AbstractVectorizer

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

    def fit(self, train_documents: Sequence[Document], train_targets: np.ndarray):
        """Train the fully connected network for all training documents."""
        # transform documents to feature vectors
        logger.info("train fully connected network with %d examples", len(train_documents))
        self.vectorizer.fit(document_as_concatenated_string(d) for d in train_documents)
        features = list(self.vectorizer.transform(document_as_concatenated_string(d) for d in train_documents))

        # make sure feature vectors is a full numpy array
        features = np.array(features)

        # convert to tensors
        features_tensor = torch.from_numpy(features).float()
        targets_tensor = torch.from_numpy(train_targets).float()

        # setup torch datatsets, such that data can be loaded in batches
        torch_dataset = TensorDataset(features_tensor, targets_tensor)
        dataloader = DataLoader(torch_dataset, batch_size=self.batch_size, shuffle=True)

        # initialize network model
        logger.info("initialize torch model on device '%s'", self.device)
        self.model = self.get_model(int(features.shape[1]), int(train_targets.shape[1]))
        self.model.to(self.device)
        self.model.train()

        # define loss and optimizer
        criterion = BCEWithLogitsLoss()
        # optimizer = SGD(self.model.parameters(), lr=self.lr)
        optimizer = Adam(self.model.parameters(), lr=self.lr)

        # iterate over epochs and batches
        for epoch in range(self.epochs):
            loss: Any = None
            batch = 0
            epoch_loss = 0
            for batch, (X, y) in enumerate(dataloader):
                # send features and targets to device
                X, y = X.to(self.device), y.to(self.device)

                # calculate loss
                output = self.model(X)
                loss = criterion(output, y)

                # do backpropagation
                optimizer.zero_grad()
                loss.backward()
                optimizer.step()

                epoch_loss += loss.item()

            epoch_loss = epoch_loss / batch
            logger.debug("trained model for epoch %d with %d batches and avg loss of %s", epoch, batch, epoch_loss)

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
        return cast(np.ndarray, cast(Any, scipy).special.expit(np.vstack(arrays)))

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
