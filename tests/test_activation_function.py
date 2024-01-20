import pytest
import torch.nn as nn
from src.classes.activation_factory import Activation_factory

def test_activation_factory_register():
    activation_factory = Activation_factory()

    activation_factory.register('sigmoid', nn.Sigmoid())

    sigmoid_activation = activation_factory('sigmoid')

    assert sigmoid_activation is not None
    assert isinstance(sigmoid_activation, nn.Sigmoid)

def test_activation_factory_register2():
    activation_factory = Activation_factory()

    activation_factory.register('relu', nn.ReLU())

    relu_activation = activation_factory('relu')

    assert relu_activation is not None
    assert isinstance(relu_activation, nn.ReLU)


if __name__ == '__main__':
    pytest.main()
