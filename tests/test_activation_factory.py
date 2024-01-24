import pytest
import torch.nn as nn
from src.classes.activation_factory import Activation_factory


@pytest.fixture
def create_activation_factory():
    return Activation_factory()


@pytest.mark.parametrize("key, function",
                        [('LeakyReLU', nn.LeakyReLU),
                         ('selu', nn.SELU)])

def test_activation_factory(create_activation_factory, key, function):
    # Arrange
    activation_factory = create_activation_factory

    # Register a new activation function
    activation_factory.register(key, function)

    # Act
    # Call the registered activation function
    leaky_relu_activation = activation_factory(key)

    # Assert
    # Check if the activation function is not None, therefore it exists
    # Check that the activation function is an instance of nn.LeakyReLU
    assert leaky_relu_activation == function

# Testing if the ValueError is being raised
def test_registrating_existing_activation_raise_error(create_activation_factory):
    # Arrange
    activation_factory = create_activation_factory

    # Assert: Use pytest.raises to check if ValueError is raised
    with pytest.raises(ValueError, match="relu activation function already exists."):
        # Act: Attempt to register the same activation function again with the correct key
        activation_factory.register('relu', nn.ReLU())


# Testing if the KeyError is being raised
def test_call_nonexistent_activation(create_activation_factory):
    # Arrange
    activation_factory = create_activation_factory

    # Act and Assert
    with pytest.raises(KeyError, match="nonexistent_activation function does not exist in dictionary."):
        activation_factory('nonexistent_activation')
