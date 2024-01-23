import pytest
import torch
from classes.neural_network import Neural_network


@pytest.mark.parametrize("type, dims, activation, lr, expected_output_size",
                         [('dense', [2, 3], 'relu', 0.0001, (1, 3)),
                          ('dense', [3, 2], 'sigmoid', 0.001, (1, 2)),
                          ('vanilla_low_rank', [2, 3], 'tanh', 0.0001, 5, (1, 3))])


def test_neural_network_parametrized(type, dims, activation, lr, rank, expected_output_size):
    # Arrange
    layer_configs = [{'type': type, 'dims': dims, 'activation': activation, 'rank': rank}]
    neural_net = Neural_network(layer_configs, lr)
    input_tensor = torch.randn(1, dims[0])  # Example input size

    # Act
    output = neural_net(input_tensor)

    # Assert
    assert output.size() == expected_output_size
