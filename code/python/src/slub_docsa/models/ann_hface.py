"""Huggingface Sequence Classification Model."""

# pylint: disable=too-many-instance-attributes, invalid-name, no-member

import logging
import os
import tempfile

from typing import List, Optional, Sequence, cast

import torch
import numpy as np
from torch.nn.modules.module import Module
from torch.utils.data import Dataset as TorchDataset
from transformers.models.auto.tokenization_auto import AutoTokenizer
from transformers.models.auto.modeling_auto import AutoModelForSequenceClassification
from transformers.training_args import TrainingArguments
from transformers.trainer import Trainer

from slub_docsa.common.paths import CACHE_DIR
from slub_docsa.common.document import Document
from slub_docsa.common.model import Model
from slub_docsa.data.preprocess.document import document_as_concatenated_string
from slub_docsa.evaluation.incidence import subject_targets_from_incidence_matrix

logger = logging.getLogger(__name__)

HUGGINGFACE_CACHE_DIR = os.path.join(CACHE_DIR, "huggingface")


class _CustomTorchDataset(TorchDataset):
    """A custom torch dataset implementation providing access to features and labels."""

    def __init__(self, encodings, labels):
        super().__init__()
        self.encodings = encodings
        self.labels = labels

    def __getitem__(self, idx):
        item = {key: torch.tensor(val[idx]) for key, val in self.encodings.items()}
        item['labels'] = torch.tensor(self.labels[idx])
        return item

    def __len__(self):
        return len(self.labels)


class HuggingfaceSequenceClassificationModel(Model):
    """A model using Huggingface models for classification.

    Uses the `AutoModelForSequenceClassification` class for training and prediction.
    """

    def __init__(
        self,
        model_identifier: str,
        epochs: int = 5,
        batch_size: int = 8,
        cache_dir: str = HUGGINGFACE_CACHE_DIR,
    ):
        """Initialize model.

        Parameters
        ----------
        model_identifier: str
            The model path string of pre-trained Huggingface models
        epochs: int
            The number of training epochs
        batch_size: int
            The number of examples processed as one batch
        cache_dir: str
            The directory used to download pre-trained Huggingface models
        """
        self.model_identifier = model_identifier
        self.epochs = epochs
        self.batch_size = batch_size
        self.n_unique_subjects = None
        self.cache_dir = cache_dir
        self.max_train_samples = 1000
        self.model: Optional[Module] = None
        self.trainer: Optional[Trainer] = None

        os.makedirs(self.cache_dir, exist_ok=True)

        logger.debug("load tokenizer")
        try:
            self.tokenizer = AutoTokenizer.from_pretrained(
                self.model_identifier,
                local_files_only=True,
                cache_dir=self.cache_dir,
            )
        except OSError:
            self.tokenizer = AutoTokenizer.from_pretrained(
                self.model_identifier,
                cache_dir=self.cache_dir,
            )

    def _tokenize_documents(self, documents: Sequence[Document], max_samples: int = None):
        corpus = [document_as_concatenated_string(d) for d in documents]
        if max_samples is not None:
            corpus = corpus[:max_samples]
        encodings = self.tokenizer(
            corpus,
            padding="max_length",
            truncation=True,
            return_tensors="pt"
        )
        return encodings

    def fit(
        self,
        train_documents: Sequence[Document],
        train_targets: np.ndarray,
        validation_documents: Optional[Sequence[Document]] = None,
        validation_targets: Optional[np.ndarray] = None,
    ):
        """Train Huggingface sequence classification network."""
        logger.debug("tokenize documents")
        train_encodings = self._tokenize_documents(train_documents, self.max_train_samples)

        logger.debug("transform labels")
        self.n_unique_subjects = int(train_targets.shape[1])
        numbered_subjects = [str(i) for i in range(self.n_unique_subjects)]
        train_labels = subject_targets_from_incidence_matrix(train_targets[:self.max_train_samples], numbered_subjects)
        train_labels = [int(next(iter(label_list))) for label_list in train_labels]

        logger.debug("load pre-trained model")
        try:
            self.model = AutoModelForSequenceClassification.from_pretrained(
                self.model_identifier,
                cache_dir=self.cache_dir,
                local_files_only=True,
                num_labels=self.n_unique_subjects
            )
        except OSError:
            self.model = AutoModelForSequenceClassification.from_pretrained(
                self.model_identifier,
                cache_dir=self.cache_dir,
                num_labels=self.n_unique_subjects
            )
        logger.debug("loaded model: %s", str(self.model))

        # disable training for bert layers
        for param in self.model.bert.parameters():
            param.requires_grad = False

        logger.debug("create torch dataset")
        tdataset = _CustomTorchDataset(train_encodings, train_labels)

        with tempfile.TemporaryDirectory() as dirname:
            training_args = TrainingArguments(
                dirname,
                per_device_train_batch_size=self.batch_size,
                num_train_epochs=self.epochs
            )
            self.trainer = Trainer(
                model=self.model, args=training_args, train_dataset=tdataset
            )
            self.trainer.train()

    def predict_proba(self, test_documents: Sequence[Document]) -> np.ndarray:
        """Predict class probabilities for trained model."""
        if self.trainer is None or self.model is None:
            raise ValueError("model has not been trained yet")

        logger.debug("tokenize %d test documents", len(test_documents))

        test_encodings = self._tokenize_documents(test_documents)
        test_labels = [0] * len(test_documents)
        tdataset = _CustomTorchDataset(test_encodings, test_labels)

        logger.debug("predict %d test documents", len(test_documents))
        result = self.trainer.predict(tdataset)
        predictions = cast(List[np.ndarray], result.predictions)

        return cast(np.ndarray, np.vstack(predictions))
