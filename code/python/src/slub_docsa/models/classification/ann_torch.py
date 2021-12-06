"""Torch Models for Classification."""

# pylint: disable=fixme, invalid-name, no-member, too-many-locals, too-many-statements, too-many-arguments
# pylint: disable=too-many-instance-attributes

import logging
import os
import pickle  # nosec

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

from slub_docsa.common.model import PersistableClassificationModel
from slub_docsa.common.document import Document
from slub_docsa.data.preprocess.document import document_as_concatenated_string
from slub_docsa.data.preprocess.vectorizer import AbstractVectorizer, PersistableVectorizer
from slub_docsa.evaluation.score import scikit_incidence_metric, scikit_metric_for_best_threshold_based_on_f1score
from slub_docsa.evaluation.incidence import positive_top_k_incidence_decision
from slub_docsa.evaluation.plotting import ann_training_history_plot, write_multiple_figure_formats

logger = logging.getLogger(__name__)

TORCH_MODEL_STATE_FILENAME = "torch_model_state.pickle"
TORCH_MODEL_SHAPE_FILENAME = "torch_model_shape.pickle"


class AbstractTorchModel(PersistableClassificationModel):
    """A abstract torch model.

    Implement the `get_model` method to provide your custom network model.
    """

    def __init__(
        self,
        vectorizer: AbstractVectorizer,
        epochs: int = 50,
        batch_size: int = 32,
        lr: float = 0.001,
        plot_training_history_filepath: str = None,
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
        self.model_shape = None
        self.batch_size = batch_size
        self.lr = lr
        self.plot_training_history_filepath = plot_training_history_filepath
        self.device = "cuda" if torch.cuda.is_available() else "cpu"

    def get_model(self, n_inputs, n_outputs) -> torch.nn.Module:
        """Return a torch network that will be trained and evaluated."""
        raise NotImplementedError()

    def _fit_epoch(self, train_dataloader, criterion, optimizer, calculate_f1_scores: bool = False):
        if self.model is None:
            raise RuntimeError("can't fit a model that is not yet initialized")

        # do training
        loss: Any = None
        batch = 0
        epoch_loss = 0
        epoch_best_threshold_f1_score = None
        epoch_top3_f1_score = None

        output_arrays = []
        y_arrays = []

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

            epoch_loss += loss.item()

            if calculate_f1_scores:
                output_arrays.append(output.cpu().detach().numpy())
                y_arrays.append(y.cpu().detach().numpy())

        if calculate_f1_scores:
            predicted_probabilities = cast(Any, scipy).special.expit(np.vstack(output_arrays))
            train_targets = cast(np.ndarray, np.vstack(y_arrays))

            epoch_best_threshold_f1_score = scikit_metric_for_best_threshold_based_on_f1score(
                f1_score, average="micro", zero_division=0
            )(train_targets, predicted_probabilities)

            epoch_top3_f1_score = scikit_incidence_metric(
                positive_top_k_incidence_decision(3), f1_score, average="micro", zero_division=0
            )(train_targets, predicted_probabilities)

        epoch_loss = epoch_loss / (batch + 1)
        return epoch_loss, epoch_best_threshold_f1_score, epoch_top3_f1_score

    def _validate_epoch(self, validation_dataloader, criterion):
        if self.model is None:
            raise RuntimeError("can't validate a model that is not yet initialized")

        # calculate test error on validation data
        epoch_loss = np.nan
        epoch_best_threshold_f1_score = np.nan
        epoch_top3_f1_score = np.nan
        if validation_dataloader is not None:
            # set model to evalution mode (not doing dropouts, etc.)
            self.model.eval()

            # get loss for validation data
            output_arrays = []
            y_arrays = []
            with torch.no_grad():
                epoch_loss = 0
                batch = 0
                for batch, (X, y) in enumerate(validation_dataloader):
                    X, y = X.to(self.device), y.to(self.device)
                    output = self.model(X)
                    loss = criterion(output, y)
                    epoch_loss += loss.item()

                    output_arrays.append(output.cpu().detach().numpy())
                    y_arrays.append(y.cpu().detach().numpy())

                epoch_loss = epoch_loss / (batch + 1)

            # compare validation outputs with true targets, and calculate f1 score
            validation_probabilities = cast(Any, scipy).special.expit(np.vstack(output_arrays))
            validation_targets = cast(np.ndarray, np.vstack(y_arrays))

            epoch_best_threshold_f1_score = scikit_metric_for_best_threshold_based_on_f1score(
                f1_score, average="micro", zero_division=0
            )(validation_targets, validation_probabilities)

            epoch_top3_f1_score = scikit_incidence_metric(
                positive_top_k_incidence_decision(3), f1_score, average="micro", zero_division=0
            )(validation_targets, validation_probabilities)

            # reset model to training mode
            self.model.train()

        return epoch_loss, epoch_best_threshold_f1_score, epoch_top3_f1_score

    def _get_data_loader_from_documents(self, texts, targets, batch_size, shuffle):
        # extract features from texts
        features = list(self.vectorizer.transform(iter(texts)))
        features = np.array(features)

        # convert to tensors
        features_tensor = torch.from_numpy(features).float()
        targets_tensor = torch.from_numpy(targets).float()

        # wrap as torch data loader
        dataset = TensorDataset(features_tensor, targets_tensor)
        dataloader = DataLoader(dataset, batch_size=batch_size, shuffle=shuffle)

        return dataloader, features.shape

    def fit(
        self,
        train_documents: Sequence[Document],
        train_targets: np.ndarray,
        validation_documents: Optional[Sequence[Document]] = None,
        validation_targets: Optional[np.ndarray] = None,
    ):
        """Train the fully connected network for all training documents."""
        logger.info("train torch network with %d training examples", len(train_documents))
        train_corpus = [document_as_concatenated_string(d) for d in train_documents]

        logger.debug("fit vectorizer based on training documents")
        self.vectorizer.fit(iter(train_corpus))

        # compile training data as data loader (and transform documents to features according to vectorizer)
        logger.debug("transform training data into tensors")
        train_dataloader, train_features_shape = self._get_data_loader_from_documents(
            texts=train_corpus,
            targets=train_targets,
            batch_size=self.batch_size,
            shuffle=True,
        )

        # same for validation data in case it is available
        validation_dataloader = None
        if validation_documents is not None and validation_targets is not None:
            logger.debug("transform validation data into tensors")
            validation_corpus = [document_as_concatenated_string(d) for d in validation_documents]
            validation_dataloader, _ = self._get_data_loader_from_documents(
                texts=validation_corpus,
                targets=validation_targets,
                batch_size=self.batch_size,
                shuffle=False,
            )

        # initialize the torch model
        logger.info("initialize torch model on device '%s'", self.device)
        self.model_shape = (int(train_features_shape[1]), int(train_targets.shape[1]))
        self.model = self.get_model(*self.model_shape)
        self.model.to(self.device)
        self.model.train()

        # define loss and optimizer
        criterion = BCEWithLogitsLoss()
        # optimizer = Adam(self.model.parameters(), lr=self.lr, weight_decay=0.0000001)
        optimizer = Adam(self.model.parameters(), lr=self.lr, weight_decay=0.0)
        scheduler = ExponentialLR(optimizer, gamma=0.99)

        epoch_train_loss_history = []
        epoch_vali_loss_history = []
        epoch_train_best_threshold_f1_score_history = []
        epoch_vali_best_threshold_f1_score_history = []
        epoch_train_top3_f1_score_history = []
        epoch_vali_top3_f1_score_history = []

        # iterate over epochs and batches
        for epoch in range(self.epochs):

            # do fit for one epoch and calculate train loss and train f1_score
            epoch_train_loss, epoch_train_best_threshold_f1_score, epoch_train_top3_f1_score = self._fit_epoch(
                train_dataloader, criterion, optimizer,
                calculate_f1_scores=validation_documents is not None
            )

            # do validation and calculate loss and f1_score
            epoch_validation_loss, epoch_vali_best_threshold_f1_score, epoch_vali_top3_f1_score = self._validate_epoch(
                validation_dataloader, criterion
            )

            # remember loss and score for each epoch
            epoch_train_loss_history.append(epoch_train_loss)
            epoch_vali_loss_history.append(epoch_validation_loss)
            epoch_train_best_threshold_f1_score_history.append(epoch_train_best_threshold_f1_score)
            epoch_vali_best_threshold_f1_score_history.append(epoch_vali_best_threshold_f1_score)
            epoch_train_top3_f1_score_history.append(epoch_train_top3_f1_score)
            epoch_vali_top3_f1_score_history.append(epoch_vali_top3_f1_score)

            logger.debug(
                "trained epoch %d, train loss %.5f, test loss %.5f, test t=best f1 %.3f, test top3 f1 %.3f",
                epoch, epoch_train_loss, epoch_validation_loss, epoch_vali_best_threshold_f1_score,
                epoch_vali_top3_f1_score
            )

            scheduler.step()
            logger.debug("adapt learning rate to %s", optimizer.param_groups[0]["lr"])

        if validation_documents is not None and self.plot_training_history_filepath:
            fig = ann_training_history_plot(
                epoch_train_loss_history,
                epoch_vali_loss_history,
                epoch_train_best_threshold_f1_score_history,
                epoch_vali_best_threshold_f1_score_history,
                epoch_train_top3_f1_score_history,
                epoch_vali_top3_f1_score_history,
            )
            write_multiple_figure_formats(
                fig, self.plot_training_history_filepath
            )

    def predict_proba(self, test_documents: Sequence[Document]) -> np.ndarray:
        """Predict class probabilities for all test documents."""
        if not self.model:
            raise ValueError("no model trained yet")

        # transform documents to feature vectors
        features = list(self.vectorizer.transform(document_as_concatenated_string(d) for d in test_documents))
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

    def save(self, persist_dir):
        """Save torch model state to disk."""
        if self.model is None or self.model_shape is None:
            raise ValueError("can not save model that was not previously fitted")

        logger.info("save pytorch model state and shape to %s", persist_dir)
        os.makedirs(persist_dir, exist_ok=True)
        torch.save(self.model.state_dict(), os.path.join(persist_dir, TORCH_MODEL_STATE_FILENAME))
        with open(os.path.join(persist_dir, TORCH_MODEL_SHAPE_FILENAME), "wb") as file:
            pickle.dump(self.model_shape, file)

        logger.info("save vectorizer to %s", persist_dir)
        if not isinstance(self.vectorizer, PersistableVectorizer):
            raise ValueError("can not save vectorizer that is not persistable")
        self.vectorizer.save(persist_dir)

    def load(self, persist_dir):
        """Load torch model state from disk."""
        if self.model is not None or self.model_shape is not None:
            raise ValueError("trying to load a persisted model after it was fitted already")

        model_state_path = os.path.join(persist_dir, TORCH_MODEL_STATE_FILENAME)
        model_shape_path = os.path.join(persist_dir, TORCH_MODEL_SHAPE_FILENAME)

        if not os.path.exists(model_state_path):
            raise ValueError(f"torch model state does not exist at {model_state_path}")

        if not os.path.exists(model_shape_path):
            raise ValueError(f"torch model shape does not exist at {model_shape_path}")

        logger.info("load torch model shape and state from files")
        with open(model_shape_path, "rb") as file:
            self.model_shape = pickle.load(file)  # nosec
        self.model = self.get_model(*self.model_shape)
        self.model.load_state_dict(torch.load(model_state_path))
        self.model.to(self.device)

        logger.info("load vectorizer from %s", persist_dir)
        if not isinstance(self.vectorizer, PersistableVectorizer):
            raise ValueError("can not load vectorizer that is not persistable")
        self.vectorizer.load(persist_dir)

    def __str__(self):
        """Return representative string for model."""
        return f"<{self.__class__.__name__} vectorizer={str(self.vectorizer)} " + \
            f"epochs={self.epochs} batch_size={self.batch_size} lr={self.lr}>"


class TorchBertSequenceClassificationHeadModel(AbstractTorchModel):
    """A torch model that follows the classification head of a Bert Sequence Classification network.

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


class TorchSingleLayerDenseReluModel(AbstractTorchModel):
    """A simple torch model consisting of one hidden layer of 1024 neurons with ReLU activations."""

    def get_model(self, n_inputs, n_outputs):
        """Return the linear network."""
        return Sequential(
            Linear(n_inputs, 1024),
            ReLU(),
            Dropout(p=0.0),
            Linear(1024, n_outputs),
        )


class TorchSingleLayerDenseTanhModel(AbstractTorchModel):
    """A simple torch model consisting of one hidden layer of 1024 neurons with tanh activations."""

    def get_model(self, n_inputs, n_outputs):
        """Return the linear network."""
        return Sequential(
            # Dropout(p=0.2),
            Linear(n_inputs, 1024),
            Tanh(),
            # Dropout(p=0.2),
            Linear(1024, n_outputs),
        )
