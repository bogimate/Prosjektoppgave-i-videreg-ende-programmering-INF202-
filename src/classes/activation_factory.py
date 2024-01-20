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
            #'sigmoid': nn.Sigmoid(),
            'softmax': nn.Softmax()
        }

    def register(self, key, name):
        """
        Registers a new activation function with a given key
        Args:
            key (str):        the name of the activation function
            name (nn.Module): The PyTorch activation function
        """

        # Checking if the activation function alredy is in _activation_type. If not, it is added
        if key not in self._function_type:
            self._function_type[key] = name
        
        else: 
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
        return self._function_type[function]


activation = Activation_factory()
#act = activation('lin')
#print(act)

activation.register('relu', nn.ReLU())
for activation_key, activation_func in activation._function_type.items():
    print(f"{activation_key}: {activation_func}")

