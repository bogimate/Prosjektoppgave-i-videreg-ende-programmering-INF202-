import pytest
import torch
import torch.nn as nn
from classes.neural_network import Neural_network


@pytest.fixture
def create_neural_network():
    def _create_neural_network(layer_configs, lr=0.0001):
        return Neural_network(layer_configs, lr)
    return _create_neural_network

@pytest.mark.parametrize("layer_configs, lr, expected_output_size", 
                      [([{'type': 'dense', 'dims': [784, 512], 'activation': 'relu'},
                         {'type': 'dense', 'dims': [512, 10], 'activation': 'softmax'},
                            ], 0.001, (1, 10)),])


def test_neural_network(create_neural_network, layer_configs, lr, expected_output_size):
    # Arrange
    neural_net = create_neural_network(layer_configs, lr)
    input_tensor = torch.randn(1, 1, 28, 28)  # Example input size

    # Act
    output = neural_net(input_tensor)

    # Assert
    assert output.size() == expected_output_size
