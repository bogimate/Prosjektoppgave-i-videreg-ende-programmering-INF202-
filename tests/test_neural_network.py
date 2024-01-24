import pytest
import torch
from src.classes.neural_network import Neural_network

@pytest.fixture
def create_neural_network():
    def _create_neural_network(layer_configs, lr):
        return Neural_network(layer_configs, lr)

    return _create_neural_network

@pytest.mark.parametrize("type, dims, activation, lr, rank, expected_output_size",
                         [('dense', [2, 3], 'relu', 0.0001, None, (1, 3)),
                          ('dense', [3, 2], 'sigmoid', 0.001, None, (1, 2)),
                          ('vanilla_low_rank', [3, 4], 'linear', 0.0001, 3, (1, 4)),
                          ('vanilla_low_rank', [4, 5], 'relu', 0.0001, 4, (1, 5))])


def test_neural_network_forward(type, dims, activation, lr, rank, expected_output_size, create_neural_network):
    # Arrange
    layer_configs = [{'type': type, 'dims': dims, 'activation': activation, 'rank': rank}]
    neural_net = create_neural_network(layer_configs, lr)
    input_tensor = torch.randn(1, dims[0])  # Example input size

    # Act
    output = neural_net(input_tensor)

    # Assert
    assert output.size() == expected_output_size
