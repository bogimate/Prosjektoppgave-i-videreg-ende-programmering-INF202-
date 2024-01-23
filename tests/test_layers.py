import pytest
import torch
import torch.nn as nn
from src.classes.layers import Dense_layer, Vanilla_low_rank_layer
from src.classes.activation_factory import Activation_factory


@pytest.fixture
def create_activation_factory():
    def _create_activation_factory(activation_type):
        return Activation_factory()(activation_type)
    return _create_activation_factory


@pytest.mark.parametrize("input_size, output_size, batch_size, activation", [
    (5, 3, 10, 'relu'),
    (8, 4, 5, 'relu'),
    (10, 5, 8, 'linear'),
])


# Test for forward pass in Dense_layer
def test_Dense_layer_forward(input_size, output_size, batch_size, activation, create_activation_factory):

    activation_function = create_activation_factory(activation)

    # Making a dense layer 
    dense_layer = Dense_layer(input_size, output_size, activation_function)

    # Generate a random input tensor
    X = torch.randn((batch_size, input_size))

    # Calling the forward function
    output = dense_layer(X)

    # Checking if the output dimentions is as expected
    assert output.shape == (batch_size, output_size)

@pytest.mark.parametrize("input_size, output_size, batch_size, learning_rate, activation", [
    (5, 3, 10, 0.001, 'relu'),
    (8, 4, 5, 0.0001, 'relu'),
    (10, 5, 8, 0.01, 'linear'),
])

# Test for update (backward pass) in Dense_layer
def test_Dense_layer_update(input_size, output_size, batch_size, learning_rate, activation, create_activation_factory):
    # Create a Dense_layer instance

    activation_function = create_activation_factory(activation)
 
    dense_layer = Dense_layer(input_size, output_size, activation_function)

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

@pytest.mark.parametrize("input_size, rank, output_size, batch_size, activation", [
    (5, 2, 3, 10, 'relu'),
    (8, 3, 4, 5, 'relu'),
    (10, 5, 5, 8, 'linear'),
])

# Test for forward pass in Vanila_low_rank_layer
def test_vanilla_low_rank_layer_forward(input_size, rank, output_size, batch_size, activation, create_activation_factory):

    activation_function = create_activation_factory(activation)

    # Making a dense layer 
    vanilla_layer = Vanilla_low_rank_layer(input_size, rank, output_size, activation_function)

    # Generate a random input tensor
    X = torch.randn((batch_size, input_size))

    # Calling the forward function
    output = vanilla_layer(X)

    # Checking if the output dimentions is as expected
    assert output.shape == (batch_size, output_size)

@pytest.mark.parametrize("input_size, rank, output_size, batch_size, learning_rate, activation", [
    (5, 2, 3, 10, 0.001, 'relu'),
    (8, 3, 4, 5, 0.0001, 'relu'),
    (10, 5, 5, 8, 0.01, 'linear'),
])

def test_Vanilla_low_rank_layer_update(input_size, rank, output_size, batch_size, learning_rate, activation, create_activation_factory):

    activation_function = create_activation_factory(activation)

    vanilla_low_rank_layer = Vanilla_low_rank_layer(input_size, rank, output_size, activation_function)

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
    vanilla_low_rank_layer.update(learning_rate)

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