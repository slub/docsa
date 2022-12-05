"""Bert model."""

import torch

from transformers.models.bert.configuration_bert import BertConfig
from transformers.models.bert.modeling_bert import BertForSequenceClassification

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
            **kwargs
        )
        self.model = BertForSequenceClassification(config)

    def forward(self, token_encodings):
        """Forward."""
        return self.model(**token_encodings).logits


class TorchBertModel(AbstractTorchModel):

    def get_model(self, n_inputs, n_outputs) -> torch.nn.Module:
        return BertModule(
            vocabulary_size=n_inputs,
            max_length=self.vectorizer.max_length,
            number_of_subjects=n_outputs,
            hidden_size=256,
            num_hidden_layers=2,
            hidden_dropout_prob=0.1,
            intermediate_size=512,
            num_attention_heads=8,
            attention_dropout_prob=0.1,
            classifier_dropout=0.1,
            type_vocab_size=1,
        )
