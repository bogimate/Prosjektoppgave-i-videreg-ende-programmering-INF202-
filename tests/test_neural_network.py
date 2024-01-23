import torch
from classes.neural_network_1 import Neural_network


def test_forward_pass():
    layer_configs = [
        {'type': 'dense', 'dims': (9, 4), 'activation': 'relu'},
        {'type': 'dense', 'dims': (4, 10), 'activation': 'sigmoid'}
    ]

    neural_net = Neural_network(layer_configs, lr=0.001)
    # Create random input tensor
    input_data = torch.randn((5, 9))

    # Perform forward pass
    output = neural_net.forward(input_data)

    # Checking if the output dimentions is as expected
    assert output.shape == (5, 10)  

# Testing the 
def test_update():
    layer_configs = [
        {'type': 'dense', 'dims': (9, 4), 'activation': 'relu'},
        {'type': 'dense', 'dims': (4, 10), 'activation': 'sigmoid'}
    ]

    neural_net = Neural_network(layer_configs, lr=0.001)

    # # Create random input tensor
    # input_data = torch.randn((5, 9))

    # # Perform forward pass
    # output = neural_net.forward(input_data)

    # Perform update
    neural_net.update()

    # Add assertions based on your expectations
    # Check if the parameters have been updated for each layer
    for layer in neural_net._layers:
        assert not torch.allclose(layer._W, layer._W_initial)  # Check if W has been updated
        assert not torch.allclose(layer._b, layer._b_initial) 