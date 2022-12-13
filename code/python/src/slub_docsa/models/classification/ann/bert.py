"""Bert model."""

# pylint: disable=too-many-arguments

from typing import Any, Mapping, Optional
import torch

from transformers.models.bert.configuration_bert import BertConfig
from transformers.models.bert.modeling_bert import BertForSequenceClassification

from slub_docsa.data.preprocess.vectorizer import AbstractSequenceVectorizer
from slub_docsa.models.classification.ann.base import AbstractTorchModel


class BertModule(torch.nn.Module):
    """Tiny bert module."""

    def __init__(
        self,
        vocabulary_size: int,
        max_length: int,
        number_of_subjects: int,
        **kwargs,
    ):
        """Initialize Bert module with various parameters.

        Parameters
        ----------
        vocabulary_size : int
            the vocubulary size of the vectorizer (e.g. Wordpiece)
        max_length : int
            the maximum sequence length of the vectorizer
        number_of_subjects : int
            the number of output neurons required
        """
        super().__init__()

        config = BertConfig(
            vocab_size=vocabulary_size,
            max_position_embeddings=max_length,
            num_labels=number_of_subjects,
            type_vocab_size=1,
            **kwargs
        )
        self.model = BertForSequenceClassification(config)

    def forward(self, token_encodings):
        """Forward that returns logits from Bert model."""
        return self.model(**token_encodings).logits


class TorchBertModel(AbstractTorchModel):
    """A model based on the BERT architecture as provided by HuggingFace."""

    def __init__(
        self,
        vectorizer: AbstractSequenceVectorizer,
        max_epochs: Optional[int] = None,
        max_training_time: Optional[int] = None,
        batch_size: int = 32,
        learning_rate: float = 0.001,
        learning_rate_decay: float = 1.0,
        positive_class_weight: float = 1.0,
        positive_class_weight_decay: float = 1.0,
        preload_vectorizations: bool = False,
        dataloader_workers: int = 8,
        plot_training_history_filepath: Optional[str] = None,
        bert_config: Mapping[str, Any] = None,
    ):
        """Initialize BERT model."""
        super().__init__(
            vectorizer=vectorizer,
            max_epochs=max_epochs,
            max_training_time=max_training_time,
            batch_size=batch_size,
            learning_rate=learning_rate,
            learning_rate_decay=learning_rate_decay,
            positive_class_weight=positive_class_weight,
            positive_class_weight_decay=positive_class_weight_decay,
            preload_vectorizations=preload_vectorizations,
            dataloader_workers=dataloader_workers,
            plot_training_history_filepath=plot_training_history_filepath
        )
        self.bert_config = bert_config if bert_config is not None else {}

    def get_model(self, n_inputs, n_outputs) -> torch.nn.Module:
        """Return the torch module describing the BERT neural network."""
        return BertModule(
            vocabulary_size=n_inputs,
            max_length=self.vectorizer.max_sequence_length(),
            number_of_subjects=n_outputs,
            **self.bert_config
        )

    def __str__(self):
        """Return representative string for model."""
        return f"<{self.__class__.__name__} vectorizer={str(self.vectorizer)} " + \
            f"epochs={self.max_epochs} batch_size={self.batch_size} lr={self.learning_rate} " + \
            f"bert_config={self.bert_config}>"
