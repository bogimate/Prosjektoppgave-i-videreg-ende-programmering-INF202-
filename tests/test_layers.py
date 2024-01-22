import pytest
import torch
import torch.nn as nn
from src.classes.layers import Dense_layer, Vanilla_low_rank_layer
from src.classes.activation_factory import Activation_factory

# Test for forward pass in Dense_layer
def test_Dense_layer_forward():
    input_size = 5
    output_size = 3
    batch_size = 10
    activation = Activation_factory()('relu')
    # Making a dense layer 
    dense_layer = Dense_layer(input_size, output_size, activation)

    # Generate a random input tensor
    X = torch.randn((batch_size, input_size))

    # Calling the forward function
    output = dense_layer(X)

    # Checking if the output dimentions is as expected
    assert output.shape == (batch_size, output_size)

def test_vanilla_low_rank_layer_forward():
    input_size = 5
    output_size = 3
    rank = 2
    batch_size = 10
    activation = Activation_factory()('relu')
    # Making a dense layer 
    vanilla_layer = Vanilla_low_rank_layer(input_size, rank, output_size, activation)

    # Generate a random input tensor
    X = torch.randn((batch_size, input_size))

    # Calling the forward function
    output = vanilla_layer(X)

    # Checking if the output dimentions is as expected
    assert output.shape == (batch_size, output_size)


def test_Dense_layer_update():
    # Create a Dense_layer instance
    input_size = 10
    output_size = 5
    batch_size = 3
    activation = Activation_factory()('relu')  # You can choose any activation function
    dense_layer = Dense_layer(input_size, output_size, activation)

    # Generate a random input tensor
    X = torch.randn((batch_size, input_size))

    # Perform forward pass
    output = dense_layer(X)

    # Perform a backward pass 
    loss = torch.sum(output)
    loss.backward()

    # Perform update
    lr = 0.001
    dense_layer.update(lr)

    # Check if the parameters have been updated
    updated_W = dense_layer._W.data
    updated_b = dense_layer._b.data

    assert (updated_W != torch.randn(input_size, output_size)).any()
    assert (updated_b != torch.randn(output_size)).any()

if __name__ == '__main__':
    pytest.main()