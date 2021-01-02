"""Implement GABERT Model for Reading Comprehension."""
from torch.nn import Module
from src.utils.mapper import configmapper


@configmapper.map("models", "gabert")
class GABERT(Module):

    """Implement GABERT.

    Methods:
        forward(x_input): Returns the output of the neural network.
    """

    def __init__(self):
        """Construct the GABERT Model."""
        super(GABERT, self).__init__()

    def forward(self, x_input):
        """
        Return the output of the GABERT for an input.

        Args:
            x_input (torch.Tensor): The input tensor to the GABERT.

        Returns:
            x_output (torch.Tensor): The output tensor for the GABERT.
        """

        return x_output
