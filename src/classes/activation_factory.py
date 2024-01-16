import torch.nn as nn

class Activation_factory:
    def __init__(self):
        self._function_type = {
            'relu': nn.ReLU(),
            'linear': nn.Identity(),
            'tanh': nn.Tanh(),
            'sigmoid': nn.Sigmoid(),
            'softmax': nn.Softmax()
        }

    # Function for adding activation functions
    def register(self, key, name):
        # Checking if the activation function is alredy in _activation_type. If not, it is added
        if key not in self._function_type:
            self._function_type[key] = name
        
        else: 
            print(f"{key} activation function already exists.")

    # Calls the activation function
    def __call__(self, function):
        return self._function_type[function]
 