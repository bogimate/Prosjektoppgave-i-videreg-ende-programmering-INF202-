import torch
import torch.nn as nn
from .layers import Dense_layer, Vanilla_low_rank_layer  
from .activation_factory import Activation_factory       

# Defining the neural network architecture
class Neural_network(nn.Module):
    def __init__(self, layer_configs, lr=0.0001):
        """
        Constructs a neural network with variable layers 
        Args:
            layer_config: a list of dictionaries, each specifying the type of layer, input size, 
            output size and activation function for that layer. 
            lr: learningrate
        """
        super(Neural_network, self).__init__()
        self._flatten = nn.Flatten()           # Resize input
        self._lr = lr                          # Default value for learningrate 

        # Create an instance of the activation class
        activate_factory = Activation_factory()
        # The register function needs an instance to work

        # Using ModuleList to store layers
        self._layers = torch.nn.Sequential()

        # Define layers based on layer_configs
        for i, config in enumerate(layer_configs):
            layer_type = config['type'] 
            input_size = config['dims'][0]
            output_size = config['dims'][1]
            activation_key = config['activation']

            # Making a dense layer
            if layer_type == 'dense':
                self._layers.add_module(name=f"{i+1}_{layer_type}_{activation_key}", module=Dense_layer(input_size, output_size, activate_factory(activation_key)))
            
            # Making a vanilla low rank layer
            elif layer_type == 'vanilla_low_rank':
                size = config['rank']
                self._layers.add_module(name=f"{i+1}_{layer_type}_{activation_key}", module=Vanilla_low_rank_layer(input_size, size, output_size, activate_factory(activation_key)))

    def forward(self, X):
        """
        Returns the output of the neural network. 
        The formula implemented is Z_k = layer_k(Z_{k-1}), where Z_0 = X.
        Args:
            X: input images or batch of input images
        Returns: 
            output neural network for given input
        """
        Z = self._flatten(X)

        # Go over and evaluate all layers in the list self._layers 
        for layer in self._layers:
            Z = layer(Z)
        
        return Z
    
    # Function for updating the layers
    def update(self):
        # Calls the update method for each layer
        for layer in self._layers:
            layer.update(self._lr)

    # Function for saving the weights and biases
    def save(self, filename=None):
        """
        Saves the models parameters.
        checkpoint: Saving the models state at a certain point in training
        state_dict: Containing all the learnable parameters of the model, weights and biases
        Args:
            filename: filename for saving models parameters  
        """
        checkpoint = {'state_dict': self._layers.state_dict()}
        torch.save(checkpoint, filename) 
        print(f'Model parameters saved to: {filename}')
