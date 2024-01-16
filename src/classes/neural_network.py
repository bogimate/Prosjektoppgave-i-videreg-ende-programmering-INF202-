import torch
import torch.nn as nn

# Define your neural network architecture. Again, you can add your own functions
class NeuralNetwork(nn.Module):
    def __init__(self, layer_configs, lr=0.0001):
        """
        Constructs a neural network with variable layers 
        Args:
            layer_config: a list of dictionaries, each specifying the type of layer, input size, 
            output size and activation function for that layer. 
            lr: learingrate
        """
        super(NeuralNetwork, self).__init__()
        self._flatten = nn.Flatten()   # Resize input
        self._lr = lr                # Default value for learingrate 

        # Dictionary of activation functions (an instance)
        Activate_factory = Activation_factory()
        # The reister function needs an instance to work
        Activate_factory.register("linear", nn.Identity())
        print(Activate_factory._function_type)

        # Using ModuleList to store layers
        self._layers = torch.nn.ModuleList()

        # Define layers based on layer_configs
        for config in layer_configs:
            layer_type = config['type'] 
            input_size = config['dims'][0]
            output_size = config['dims'][1]
            activation_key = config['activation']

            if layer_type == 'dense':
                self._layers.append(Dense_layer(input_size, output_size, Activate_factory(activation_key)))
            
            elif layer_type == 'vanilla_low_rank':
                size = config['rank']
                self._layers.append(Vanilla_low_rank_layer(input_size, size, output_size, Activate_factory(activation_key)))

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

        # TODO: Go over and evaluate all layers in the list self.layers 
        for layer in self._layers:
            Z = layer(Z)
        
        return Z
    
    def update(self):
        for layer in self._layers:
            layer.update(self._lr)
        
