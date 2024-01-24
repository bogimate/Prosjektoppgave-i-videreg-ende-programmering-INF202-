import torch
import torch.nn as nn

# Dense layer
class Dense_layer(nn.Module):
    def __init__(self, input_size, output_size, activation):
        super(Dense_layer, self).__init__()
       
        # Defining the parameters
        self._W = nn.Parameter(torch.randn(input_size, output_size), requires_grad=True)
        self._b = nn.Parameter(torch.randn(output_size), requires_grad=True) 
        self.activation = activation # Assigning the activation function 
 
    def forward(self, X):
        """
        Returns the output of the layer. 
            Args:    X - input to layer
            Returns: output of layer
        """
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
class Vanilla_low_rank_layer(nn.Module):
    def __init__(self, input_size, size, output_size, activation):
        super(Vanilla_low_rank_layer, self).__init__()

        # Defining the parameters
        self._U = nn.Parameter(torch.randn(input_size, size), requires_grad=True)
        self._S = nn.Parameter(torch.randn(size, size), requires_grad=True)
        self._V = nn.Parameter(torch.randn(output_size, size), requires_grad=True)
        self._b = nn.Parameter(torch.randn(output_size), requires_grad=True) 
        self.activation = activation # Assigning the activation function

        # Ensure orthogonality
        self.orthogonalize()

    def forward(self, X):
        """
        Returns the output of the layer. 
            Args:    X - input to layer
            Returns: output of layer
        """

        # Perform linear transformation
        step_1 = torch.matmul(X, self._U)
        step_2 = torch.matmul(step_1, self._S)
        step_3 = torch.matmul(step_2, self._V.T)
        output = step_3 + self._b
        
        # Apply activation function
        activated_output = self.activation(output)
        return activated_output

    # Function to update the parameters 
    def update(self, lr):
        self._U.data = self._U - lr * self._U.grad
        self._S.data = self._S - lr * self._S.grad
        self._V.data = self._V - lr * self._V.grad
        self._b.data = self._b - lr * self._b.grad

        self._U.grad.zero_()
        self._S.grad.zero_()
        self._V.grad.zero_()
        self._b.grad.zero_()

    def orthogonalize(self):
        # Applying QR-decomposition to make U and V orthogonal
        self._U.data, _ = torch.linalg.qr(self._U.data, 'reduced')
        self._V.data, _ = torch.linalg.qr(self._V.data, 'reduced')
