import torch
import pytest
from src.classes.neural_network import Neural_network


def test_neural_network_forward():
    batch_size = 10
    input_size = 5
    # Arrange
    layer_configs = [
        {'type': 'dense', 'dims': (5, 3), 'activation': 'relu'},
        {'type': 'vanilla_low_rank', 'dims': (3, 2), 'activation': 'linear', 'rank': 2},
    ]
    neural_network = Neural_network(layer_configs)

    # Create a sample input tensor
    input_tensor = torch.randn((batch_size, input_size))

    # Act
    output_tensor = neural_network(input_tensor)

    # Assert
    # Add assertions based on the expected output shape or other properties
    assert output_tensor.shape == (10, 2)