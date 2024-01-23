import pytest
import torch
import torch.nn as nn
from src.classes.layers import Dense_layer, Vanilla_low_rank_layer
from src.classes.activation_factory import Activation_factory

@pytest.fixture
def create_activation_factory():
    return Activation_factory()

@pytest.mark.parametrize("input_size, output_size, batch_size, learning_rate", [
    (5, 3, 10, 0.001),
    (8, 4, 5, 0.0001),
    (10, 5, 8, 0.01),
])

@pytest.mark.parametrize("key, function",
                        ['relu',
                         'linear'])


# Test for forward pass in Dense_layer
def test_Dense_layer_forward(input_size, output_size, batch_size, key, create_activation_factory):

    activation_factory = create_activation_factory

    activation = activation_factory(key)

    # Making a dense layer 
    dense_layer = Dense_layer(input_size, output_size, activation)

    # Generate a random input tensor
    X = torch.randn((batch_size, input_size))

    # Calling the forward function
    output = dense_layer(X)

    # Checking if the output dimentions is as expected
    assert output.shape == (batch_size, output_size)

# Test for update (backward pass) in Dense_layer
def test_Dense_layer_update(input_size, output_size, batch_size, learning_rate, key, function, create_activation_factory):
    # Create a Dense_layer instance

    activation_factory = create_activation_factory

    activation = activation_factory(key)
 
    dense_layer = Dense_layer(input_size, output_size, activation)

    # Generate a random input tensor
    X = torch.randn((batch_size, input_size))

    # Perform forward pass to retrive output tensor
    output = dense_layer(X)

    # Store the initial values
    initial_W = dense_layer._W.data.clone()
    initial_b = dense_layer._b.data.clone()

    # Perform a backward pass 
    loss = torch.sum(output)
    # Calculates gradients with respect to the parameters (_W & _b)
    loss.backward()

    # Perform update
    dense_layer.update(learning_rate)

    # Updating the parameters
    updated_W = dense_layer._W.data
    updated_b = dense_layer._b.data

    # Check if the parameters have been updated using torch.allclose
    assert not torch.allclose(updated_W, initial_W)
    assert not torch.allclose(updated_b, initial_b)



# Test for forward pass in Vanila_low_rank_layer
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


def test_Vanilla_low_rank_layer_update():
    # Arrange
    input_size = 5
    output_size = 3
    rank = 2
    batch_size = 10
    activation = Activation_factory()('relu')  
    vanilla_low_rank_layer = Vanilla_low_rank_layer(input_size, rank, output_size, activation)

    # Generate a random input tensor
    X = torch.randn((batch_size, input_size))

    # Perform forward pass
    output = vanilla_low_rank_layer(X)

    # Store the initial values
    initial_U = vanilla_low_rank_layer._U.data.clone()
    initial_S = vanilla_low_rank_layer._S.data.clone()
    initial_V = vanilla_low_rank_layer._V.data.clone()
    initial_b = vanilla_low_rank_layer._b.data.clone()

    # Perform a backward pass 
    loss = torch.sum(output)
    loss.backward()

    # Perform update
    lr = 0.001
    vanilla_low_rank_layer.update(lr)

    # Check if the parameters have been updated
    updated_U = vanilla_low_rank_layer._U.data
    updated_S = vanilla_low_rank_layer._S.data
    updated_V = vanilla_low_rank_layer._V.data
    updated_b = vanilla_low_rank_layer._b.data

    # Check if the parameters have been updated using torch.allclose
    assert not torch.allclose(updated_U, initial_U)
    assert not torch.allclose(updated_S, initial_S)
    assert not torch.allclose(updated_V, initial_V)
    assert not torch.allclose(updated_b, initial_b)


if __name__ == '__main__':
    pytest.main()