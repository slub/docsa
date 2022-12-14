"""Torch Models for Classification."""

# pylint: disable=fixme, invalid-name, no-member, too-many-locals, too-many-statements, too-many-arguments
# pylint: disable=too-many-instance-attributes, too-few-public-methods

import logging
import os
import pickle  # nosec
import time

from typing import Any, Optional, Sequence

import numpy as np
import torch
import transformers

from torch.utils.data import DataLoader, IterableDataset, Dataset as TorchDataset
from torch.nn import BCEWithLogitsLoss
from torch.optim import Adam
from torch.optim.lr_scheduler import ExponentialLR

from slub_docsa.common.model import PersistableClassificationModel
from slub_docsa.common.document import Document
from slub_docsa.data.preprocess.document import document_as_concatenated_string
from slub_docsa.data.preprocess.vectorizer import AbstractVectorizer, PersistableVectorizerMixin
from slub_docsa.evaluation.classification.score.ann import TorchF1Score
from slub_docsa.evaluation.classification.plotting import ann_training_history_plot, write_multiple_figure_formats

logger = logging.getLogger(__name__)

TORCH_MODEL_STATE_FILENAME = "torch_model_state.pickle"
TORCH_MODEL_SHAPE_FILENAME = "torch_model_shape.pickle"


class SimpleIterableDataset(IterableDataset):
    """Simple torch dataset that provides access to feature and incidences via an iterator."""

    def __init__(self, features, subject_incidences: Optional[np.ndarray] = None):
        """Initialize dataset with iterable features and iterable subject incidences."""
        self.features = features
        self.subject_incidences = subject_incidences

    def __iter__(self):
        """Return an iterator over tuples of features and the corresponding subject incidence vector."""
        if self.subject_incidences is not None:
            return iter(zip(self.features, self.subject_incidences))
        return iter(self.features)

    def __getitem__(self, idx) -> Any:
        """Not available for iterable datasets."""
        raise NotImplementedError()


class SimpleIndexedDataset(TorchDataset):
    """Simple torch dataset that provides access to feature and incidences via an index."""

    def __init__(self, features: Sequence[Any], subjects: Optional[np.ndarray] = None):
        """Initialize dataset with a sequence of features and subject incidences."""
        self.features = features
        self.subjects = subjects

    def __getitem__(self, idx):
        """Return a specific tuple of feature and subject incidence vector at index position."""
        if self.subjects is not None:
            return self.features[idx], self.subjects[idx]
        return self.features[idx]

    def __len__(self):
        """Return the number of samples."""
        return len(self.features)


class LazyIndexedDataset(TorchDataset):
    """A torch dataset implementation that applies the vectorizer only when a sample is requested."""

    def __init__(
        self,
        vectorizer: AbstractVectorizer,
        documents: Sequence[Document],
        subject_targets: Optional[Sequence[np.ndarray]] = None,
        max_document_length: Optional[int] = 10000,
        preload: bool = False,
    ):
        """Initialize dataset with a sequence of features and subject incidences."""
        self.vectorizer = vectorizer
        self.documents = documents
        self.subject_targets = subject_targets
        self.max_document_length = max_document_length
        self.features = None
        if preload:
            logger.info("preload features by applying vectorizer to all documents")
            text_iterator = (
                document_as_concatenated_string(d, max_length=self.max_document_length) for d in self.documents
            )
            self.features = list(self.vectorizer.transform(text_iterator))

    def __getitem__(self, idx):
        """Return a specific tuple of feature and subject incidence vector at index position."""
        if self.features:
            features = self.features[idx]
        else:
            text_iterator = iter([
                document_as_concatenated_string(self.documents[idx], max_length=self.max_document_length)
            ])
            features = next(self.vectorizer.transform(text_iterator))
        if self.subject_targets is not None:
            return features, self.subject_targets[idx]
        return features

    def __len__(self):
        """Return the number of samples."""
        return len(self.documents)


class ProgressLogger:
    """Helper class to print log message during training and testing."""

    def __init__(self, stage: str, batch_size: int, score: TorchF1Score):
        """Initialize progress logger."""
        self.batch_size = batch_size
        self.last_log_time = time.time()
        self.last_log_batch = 0
        self.stage = stage
        self.score = score

    def log(self, current_batch, cumulative_epoch_loss, end_of_epoch):
        """Print log messages."""
        passed_time = time.time() - self.last_log_time
        if passed_time > 5.0 or end_of_epoch:
            samples_per_second = ((current_batch - self.last_log_batch) * self.batch_size) / passed_time
            current_avg_loss = cumulative_epoch_loss / (current_batch + 1)
            t01_f1, t05_f1 = self.score()

            # print logs
            if end_of_epoch:
                logger.info("-----------------------------------------------------")
                logger.info(
                    "processed all %s %s batches with %d samples/s, loss=%.7f",
                    current_batch, self.stage, samples_per_second, current_avg_loss
                )
            else:
                logger.info(
                    "processed %s batch %s with %d samples/s, loss=%.7f",
                    self.stage, current_batch, samples_per_second, current_avg_loss
                )

            logger.info("f1_t=0.1 total %.4f, distriubution %s", t01_f1[0], t01_f1[1])
            logger.info("f1_t=0.5 total %.4f, distriubution %s", t05_f1[0], t05_f1[1])

            # update time
            self.last_log_time = time.time()
            self.last_log_batch = current_batch


class AbstractTorchModel(PersistableClassificationModel):
    """A abstract torch model.

    Implement the `get_model` method to provide your custom network model.
    """

    def __init__(
        self,
        vectorizer: AbstractVectorizer,
        max_epochs: Optional[int] = None,
        max_training_time: Optional[int] = None,
        max_training_t05_f1: Optional[float] = None,
        batch_size: int = 32,
        learning_rate: float = 0.001,
        learning_rate_decay: float = 1.0,
        positive_class_weight: float = 1.0,
        positive_class_weight_min: float = 1.0,
        positive_class_weight_decay: float = 1.0,
        max_document_length: Optional[int] = 10000,
        preload_vectorizations: bool = True,
        dataloader_workers: int = 0,
        plot_training_history_filepath: Optional[str] = None,
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
        positive_class_weight: float = 1.0
            the positive class weight that is assigned to all positive cases (1.0 means no extra weight)
        positive_class_weight_decay: float = 1.0
            the decay factor that is applied to the positive class weight on each training epoch (1.0 means no decay)
        plot_training_history_filepath: Optional[str] = None
            the path to the training history plot
        """
        self.vectorizer = vectorizer
        self.max_epochs = max_epochs
        self.max_training_time = max_training_time
        self.max_training_t05_f1 = max_training_t05_f1
        self.model = None
        self.model_shape = None
        self.batch_size = batch_size
        self.learning_rate = learning_rate
        self.learning_rate_decay = learning_rate_decay
        self.positive_class_weight = positive_class_weight
        self.positive_class_weight_min = positive_class_weight_min
        self.positive_class_weight_decay = positive_class_weight_decay
        self.max_document_length = max_document_length
        self.preload_vectorizations = preload_vectorizations
        self.dataloader_workers = dataloader_workers
        self.plot_training_history_filepath = plot_training_history_filepath
        self.device = "cuda" if torch.cuda.is_available() else "cpu"

    def get_model(self, n_inputs, n_outputs) -> torch.nn.Module:
        """Return a torch network that will be trained and evaluated."""
        raise NotImplementedError()

    def _update_positive_class_weight(self, epoch, criterion):
        if self.positive_class_weight != 1.0 or self.positive_class_weight_decay != 1.0:
            current_weight = max(
                self.positive_class_weight_min,
                self.positive_class_weight * np.power(self.positive_class_weight_decay, epoch)
            )
            logger.info("set positive class weight to %f", current_weight)
            criterion.pos_weight = (torch.ones([self.model_shape[1]]) * current_weight).to(self.device)

    def _fit_epoch(self, epoch, train_dataloader, criterion, optimizer):
        if self.model is None:
            raise RuntimeError("can't fit a model that is not yet initialized")

        batch = 0
        cumulative_epoch_loss = 0
        epoch_score = TorchF1Score()
        progress = ProgressLogger("training", self.batch_size, epoch_score)

        self._update_positive_class_weight(epoch, criterion)

        for batch, (X, y) in enumerate(train_dataloader):
            # send features and targets to device
            if isinstance(X, transformers.BatchEncoding):
                X = X.to(self.device)
            else:
                X = X.to(device=self.device, dtype=torch.float32)
            y = y.to(self.device).to(torch.float32)

            # evaluate model
            output = self.model(X)
            loss = criterion(output, y)

            # score predicted probabilities
            predicted_probabilities = torch.special.expit(output)
            epoch_score.add_batch(y, predicted_probabilities)

            # do backpropagation
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            # remember binary cross entropy loss
            cumulative_epoch_loss += loss.item()

            progress.log(batch, cumulative_epoch_loss, False)

        progress.log(batch, cumulative_epoch_loss, True)
        epoch_loss = cumulative_epoch_loss / (batch + 1)

        t01_f1, t05_f1 = epoch_score()
        return epoch_loss, t01_f1[0], t05_f1[0]

    def _validate_epoch(self, validation_dataloader, criterion):
        if self.model is None:
            raise RuntimeError("can't validate a model that is not yet initialized")

        # calculate test error on validation data
        cumulative_epoch_loss = np.nan
        epoch_score = TorchF1Score()
        progress = ProgressLogger("validation", self.batch_size, epoch_score)

        # set model to evalution mode (not doing dropouts, etc.)
        self.model.eval()

        # get loss for validation data
        with torch.no_grad():
            cumulative_epoch_loss = 0
            batch = 0
            for batch, (X, y) in enumerate(validation_dataloader):
                # send features and targets to device
                if isinstance(X, transformers.BatchEncoding):
                    X = X.to(self.device)
                else:
                    X = X.to(device=self.device, dtype=torch.float32)
                y = y.to(self.device).to(torch.float32)

                # evaluate model
                output = self.model(X)
                loss = criterion(output, y)
                cumulative_epoch_loss += loss.item()

                # score predicted probabilities
                predicted_probabilities = torch.special.expit(output)
                epoch_score.add_batch(y, predicted_probabilities)

                progress.log(batch, cumulative_epoch_loss, False)

        progress.log(batch, cumulative_epoch_loss, True)
        epoch_loss = cumulative_epoch_loss / (batch + 1)

        # reset model to training mode
        self.model.train()

        t01_f1, t05_f1 = epoch_score()
        return epoch_loss, t01_f1[0], t05_f1[0]

    def _get_data_loader_from_documents(self, documents, targets, batch_size, shuffle):
        dataset = LazyIndexedDataset(
            self.vectorizer, documents, targets, self.max_document_length, self.preload_vectorizations
        )
        dataloader = DataLoader(dataset, batch_size=batch_size, shuffle=shuffle, num_workers=self.dataloader_workers)
        return dataloader

    def _plot_score_history(self, epoch_train_score_history, epoch_validation_score_history):
        if self.plot_training_history_filepath:
            fig = ann_training_history_plot(
                epoch_train_score_history,
                epoch_validation_score_history,
            )
            write_multiple_figure_formats(
                fig, self.plot_training_history_filepath
            )

    def fit(
        self,
        train_documents: Sequence[Document],
        train_targets: Sequence[Sequence[int]],
        validation_documents: Optional[Sequence[Document]] = None,
        validation_targets: Optional[Sequence[Sequence[int]]] = None,
    ):
        """Train the fully connected network for all training documents."""
        logger.info("train torch network with %d training examples", len(train_documents))

        logger.debug("fit vectorizer based on training documents")
        self.vectorizer.fit(
            document_as_concatenated_string(d, max_length=self.max_document_length) for d in train_documents
        )

        # compile training data as data loader (and transform documents to features according to vectorizer)
        logger.debug("transform training data into tensors")
        train_dataloader = self._get_data_loader_from_documents(
            documents=train_documents,
            targets=train_targets,
            batch_size=self.batch_size,
            shuffle=True,
        )

        # same for validation data in case it is available
        validation_dataloader = None
        if validation_documents is not None and validation_targets is not None:
            logger.debug("transform validation data into tensors")
            validation_dataloader = self._get_data_loader_from_documents(
                documents=validation_documents,
                targets=validation_targets,
                batch_size=self.batch_size,
                shuffle=False,
            )

        # initialize the torch model
        logger.info("initialize torch model on device '%s'", self.device)
        self.model_shape = (int(self.vectorizer.output_shape()[0]), int(len(train_targets[0])))
        logger.debug("model shape will be %s", str(self.model_shape))
        self.model = self.get_model(*self.model_shape)

        number_of_parameters = sum(p.numel() for p in self.model.parameters() if p.requires_grad)
        logger.info("torch model has %d parameters", number_of_parameters)

        self.model.to(self.device)
        self.model.train()

        # define loss and optimizer
        criterion = BCEWithLogitsLoss()
        # optimizer = Adam(self.model.parameters(), lr=self.lr, weight_decay=0.0000001)
        optimizer = Adam(self.model.parameters(), lr=self.learning_rate, weight_decay=0.0)
        scheduler = ExponentialLR(optimizer, gamma=self.learning_rate_decay)

        epoch_train_score_history = []
        epoch_validation_score_history = []
        last_plot_time = time.time()

        # iterate over epochs and batches
        epoch = 0
        training_started = time.time()
        while True:
            if self.max_epochs is not None and epoch >= self.max_epochs:
                logger.info("stop training because of maximum epochs")
                break

            if self.max_training_time is not None and time.time() - training_started > self.max_training_time:
                logger.info("stop training after max training time reached")
                break

            if self.max_training_t05_f1 is not None and epoch_train_score_history and \
                    epoch_train_score_history[-1][2] >= self.max_training_t05_f1:
                logger.info("stop training due to overfitting on training data")
                break

            # do fit for one epoch and calculate train loss and train f1_score
            epoch_train_score_history.append(self._fit_epoch(
                epoch, train_dataloader, criterion, optimizer,
            ))

            # do validation and calculate loss and f1_score
            if validation_dataloader is not None:
                epoch_validation_score_history.append(self._validate_epoch(
                    validation_dataloader, criterion
                ))
            else:
                epoch_validation_score_history.append((np.nan, np.nan, np.nan))

            logger.info(
                "trained epoch %d, train loss %.5f, validation loss %.5f, t=0.1 f1 %.3f, t=0.5 f1 %.3f",
                epoch, epoch_train_score_history[-1][0], *epoch_validation_score_history[-1]
            )

            if self.learning_rate_decay < 1.0:
                scheduler.step()
                logger.debug("adapt learning rate to %s", optimizer.param_groups[0]["lr"])

            if time.time() - last_plot_time > 10:
                # plot score history sometimes during training
                self._plot_score_history(epoch_train_score_history, epoch_validation_score_history)
                last_plot_time = time.time()

            epoch += 1
        # definitely plot score history after training has finished
        self._plot_score_history(epoch_train_score_history, epoch_validation_score_history)

    def predict_proba(self, test_documents: Sequence[Document]) -> np.ndarray:
        """Predict class probabilities for all test documents."""
        if not self.model:
            raise ValueError("no model trained yet")

        # transform documents to feature vectors
        features = list(self.vectorizer.transform(
            document_as_concatenated_string(d, max_length=self.max_document_length) for d in test_documents
        ))

        # setup torch datatsets
        torch_dataset = SimpleIterableDataset(features)
        dataloader = DataLoader(torch_dataset, batch_size=self.batch_size)

        # iterate over batches of all examples
        arrays = []
        self.model.eval()
        with torch.no_grad():
            for X in dataloader:
                # send features and targets to device
                if isinstance(X, transformers.BatchEncoding):
                    X = X.to(self.device)
                else:
                    X = X.to(device=self.device, dtype=torch.float32)
                # evaluate model for each test example
                outputs = self.model(X)
                probabilities = torch.special.expit(outputs)
                # retrieve outputs and collected them as numpy arrays
                arrays.append(probabilities.cpu().detach().numpy())

        # reverse logits and return results
        predictions = np.vstack(arrays)
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
        if not isinstance(self.vectorizer, PersistableVectorizerMixin):
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
        self.model.load_state_dict(torch.load(model_state_path, map_location=torch.device(self.device)))
        self.model.to(self.device)

        logger.info("load vectorizer from %s", persist_dir)
        if not isinstance(self.vectorizer, PersistableVectorizerMixin):
            raise ValueError("can not load vectorizer that is not persistable")
        self.vectorizer.load(persist_dir)

    def __str__(self):
        """Return representative string for model."""
        return f"<{self.__class__.__name__} vectorizer={str(self.vectorizer)} " + \
            f"max_epochs={self.max_epochs} batch_size={self.batch_size} lr={self.learning_rate}>"
