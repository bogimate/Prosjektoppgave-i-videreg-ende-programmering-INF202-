import torch.nn as nn


class Activation_factory:
    """
    A class for managing and providing access to various activation functions.
    """

    def __init__(self):
        self._function_type = {
            'relu': nn.ReLU(),
            'linear': nn.Identity(),
            'tanh': nn.Tanh(),
            'sigmoid': nn.Sigmoid(),
            'softmax': nn.Softmax()
        }

    def register(self, key, name):
        """
        Registers a new activation function with a given key
        Args:
            key (str):        the name of the activation function
            name (nn.Module): The PyTorch activation function
        """

        # Checking if the activation function already is in _activation_type. If not, it is added
        if key not in self._function_type:
            self._function_type[key] = name
        
        else: 
            # Raises a value error when function already excists in _function_type
            raise ValueError(f"{key} activation function already exists.")


    # Calls the activation function
    def __call__(self, function):
        """
        Retrieves and returns the activation function associated with the provided key.
        Args:
            function (str): The key representing the activation function.
        Returns:
            torch.nn.Module: The PyTorch activation function.
        """
        # If you try calling a function that is not in _function_type a KeyError will be raised
        if function not in self._function_type:
            raise KeyError(f"{function} function does not exist in dictionary.")
        else:
            # Returns the function type if its in the dictionary
            return self._function_type[function]
        





