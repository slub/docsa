"""Simple fully-connected artificial neural networks."""

from torch.nn.modules.activation import ReLU, Tanh
from torch.nn import Sequential, Linear, Dropout

from slub_docsa.models.classification.ann.base import AbstractTorchModel


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
