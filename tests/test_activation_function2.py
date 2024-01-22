import pytest
import torch.nn as nn
from src.classes.activation_factory import Activation_factory

def test_activation_factory():
    # Arrange
    activation_factory = Activation_factory()

    # Register a new activation function
    activation_factory.register('LeakyReLU', nn.LeakyReLU())

    # Act
    # Call the registered activation function
    leaky_relu_activation = activation_factory('LeakyReLU')

    # Assert
    # Check if the activation function is not None, therefore it exists
    assert leaky_relu_activation is not None
    # Check that the activation function is an instance of nn.LeakyReLU
    assert isinstance(leaky_relu_activation, nn.LeakyReLU)

def test_register_existing_activation():
    # Arrange
    activation_factory = Activation_factory()

    # Act: Register an activation function
    activation_factory.register('relu', nn.ReLU())

    # Assert: Use pytest.raises to check if ValueError is raised
    with pytest.raises(ValueError, match="relu activation function already exists."):
        activation_factory.register('relu', nn.ReLU())

def test_call_nonexistent_activation():
    # Arrange
    activation_factory = Activation_factory()

    # Act and Assert
    with pytest.raises(KeyError, match="nonexistent_activation function does not exist in dictionary."):
        activation_factory('nonexistent_activation')