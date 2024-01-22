import torch
import pytest
from src.classes.neural_network import Neural_network


def test_neural_network_forward():
    batch_size = 10
    
    layer_configs = [
        {'type': 'dense', 'input_size': 5, 'output_size': 3, 'activation': 'relu'},
        {'type': 'vanilla_low_rank', 'input_size': 3, 'rank': 2, 'output_size': 2, 'activation': 'lin'}
    ]

    neural_network = Neural_network(layer_configs)

    # Create a sample input tensor
    input_tensor = torch.randn((batch_size, layer_configs[0]['input_size']))

    # Act
    output_tensor = neural_network(input_tensor)

    # Assert
    # Add assertions based on the expected output shape or other properties
    assert output_tensor.shape == (batch_size, layer_configs[-1]['output_size'])
