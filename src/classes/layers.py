import torch
import torch.nn as nn

# Dense layer
class Dense_layer(nn.Module):
    def __init__(self, inputSize, outputSize, activation):
        super(Dense_layer, self).__init__()
       
        # Defining the parameters
        self._W = nn.Parameter(torch.randn(inputSize, outputSize), requires_grad=True)
        self._b = nn.Parameter(torch.randn(outputSize), requires_grad=True) 
        self.activation = activation # Assigning the activation function 
 
    def forward(self, X):
        # Returns the output of the layer. 
            # Args:    X - input to layer
            # Returns: output of layer

        # Perform linear transformation
        output = torch.matmul(X, self._W) + self._b
        # Apply activation function
        activated_output = self.activation(output)
        return activated_output

    # Function to update the parameters 
    def update(self, lr):
        self._W.data = self._W - lr * self._W.grad
        self._b.data = self._b - lr * self._b.grad

        self._W.grad.zero_()
        self._b.grad.zero_()


# Vanilla low rank layer