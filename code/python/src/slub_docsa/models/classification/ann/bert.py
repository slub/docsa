"""Bert model."""

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
        """Forward."""
        return self.model(**token_encodings).logits


class TorchBertModel(AbstractTorchModel):

    def __init__(
        self,
        vectorizer: AbstractSequenceVectorizer,
        epochs: int = 50,
        batch_size: int = 32,
        lr: float = 0.001,
        plot_training_history_filepath: Optional[str] = None,
        bert_config: Mapping[str, Any] = None,
    ):
        """Initialize custom Bert model."""
        super().__init__(vectorizer, epochs, batch_size, lr, plot_training_history_filepath)
        self.bert_config = bert_config if bert_config is not None else {}

    def get_model(self, n_inputs, n_outputs) -> torch.nn.Module:
        return BertModule(
            vocabulary_size=n_inputs,
            max_length=self.vectorizer.max_sequence_length(),
            number_of_subjects=n_outputs,
            **self.bert_config
        )

    def __str__(self):
        """Return representative string for model."""
        return f"<{self.__class__.__name__} vectorizer={str(self.vectorizer)} " + \
            f"epochs={self.epochs} batch_size={self.batch_size} lr={self.lr} bert_config={self.bert_config}>"
