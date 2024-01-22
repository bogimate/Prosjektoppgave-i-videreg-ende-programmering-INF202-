import torch
from src.classes.neural_network import Neural_network

@pytest.fixture
def neural_network():
    # Create a Neural_network instance for testing
    layer_configs = [
        {'type': 'dense', 'dims': (9, 4), 'activation': 'relu'},
        {'type': 'dense', 'dims': (4, 10), 'activation': 'sigmoid'}
    ]
    return Neural_network(layer_configs, lr=0.001)

# Testing the forward pass of the neural network
def test_forward_pass(neural_network):
    # Create random input tensor
    input_data = torch.randn((5, 9))

    # Perform forward pass
    output = neural_network.forward(input_data)

    # Checking if the output dimentions is as expected
    assert output.shape == (5, 10)  

# Testing the 
def test_update(neural_network):
    # Perform update
    neural_network.update()

    # Add assertions based on your expectations
    # Check if the parameters have been updated for each layer
    for layer in neural_network._layers:
        assert not torch.allclose(layer._W, layer._W_initial)  # Check if W has been updated
        assert not torch.allclose(layer._b, layer._b_initial)