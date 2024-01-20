import pytest
import torch.nn as nn
from src.classes.activation_factory import Activation_factory

def test_activation_factory():
    activation_factory = Activation_factory()

    # Register a new activation function 
    activation_factory.register('sigmoid', nn.Sigmoid())

    # Call the registered activation function
    sigmoid_activation = activation_factory('sigmoid')

    # Checks if the activation function is not None, therefore it excists 
    assert sigmoid_activation is not None
    # Check that the activation function is an instence of nn.ReLU
    assert isinstance(sigmoid_activation, nn.Sigmoid)

# Testing the factory to see that it fails
# Trying to add an already existing activation function
def test_activation_factory_failing():
    activation_factory = Activation_factory()

    activation_factory.register('relu', nn.ReLU())

    relu_activation = activation_factory('relu')

    assert relu_activation is not None
    assert isinstance(relu_activation, nn.ReLU)


if __name__ == '__main__':
    pytest.main()

