import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from torchvision import datasets, transforms
import toml
#from class_Read_config import Read_config

# Final Validation Accuracy 88.83%
# The best working code at the moment without orthonormal U and V
# With config read function

class Activation_factory:
    def __init__(self):
        self._function_type = {
            'relu': nn.ReLU(),
            #'linear': nn.Identity(),
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
 

# A class for creating layers
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

class Vanilla_low_rank_layer(nn.Module):
    def __init__(self, inputSize, Size, outputSize, activation):
        super(Vanilla_low_rank_layer, self).__init__()

        # Defining the parameters
        self._U = nn.Parameter(torch.randn(inputSize, Size), requires_grad=True)
        self._S = nn.Parameter(torch.randn(Size, Size), requires_grad=True)
        self._V = nn.Parameter(torch.randn(Size, outputSize), requires_grad=True)
        self._b = nn.Parameter(torch.randn(outputSize), requires_grad=True) 
        self.activation = activation # Assigning the activation function

    def forward(self, X):
        # Returns the output of the layer. 
            # Args:    X - input to layer
            # Returns: output of layer

        # Perform linear transformation
        step_1 = torch.matmul(X, self._U)
        step_2 = torch.matmul(step_1, self._S)
        step_3 = torch.matmul(step_2, self._V)
        output = step_3 + self._b

        # output = torch.matmul(torch.matmul(torch.matmul(X, self._U), self._S), self._V) + self._b
        
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

def read_config(name):
    with open(name, 'r') as file:
        conf = toml.load(file)
    
    return conf.get('settings'), conf.get("layer")

config_settings, config_layers = read_config('our_input.toml')

lr = config_settings.get('learningRate', 0.0001)
batchSize = config_settings.get('batchSize', 64)
num_epochs = config_settings.get('numEpochs', 10)

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
        


# Load MNIST dataset
transform = transforms.Compose([transforms.ToTensor(), transforms.Normalize((0.5,), (0.5,))])

traindataset = datasets.MNIST(root='./data', train=True, transform=transform, download=True)
testdataset  = datasets.MNIST(root='./data', train=False, transform=transform, download=True)

trainloader = DataLoader(traindataset, batch_size=batchSize, shuffle=True)
testloader  = DataLoader(testdataset, batch_size=batchSize, shuffle=False)



criterion = nn.CrossEntropyLoss()
# Create an instance of NeuralNetwork and directly use the activation functions from the Activation class
NeuralNet = NeuralNetwork(config_layers, lr)

# Training loop
for i in range(num_epochs):
    for step, (images, labels) in enumerate(trainloader):
        out = NeuralNet(images)

        loss = criterion(out, labels)
        loss.backward()

        NeuralNet.update()

        # TODO: Train your weights and biases. Think about how you can use object oriented programming to do so.
        # The way you do this heavily impacts the extendability of your code. So take some time to think about the design of your program!

        if (step + 1) % 100 == 0:
            print(f'Epoch [{i+1}/{num_epochs}], Step [{step+1}/{len(trainloader)}], Loss: {loss.item():.4f}')

    correct = 0
    total = 0
    with torch.no_grad():
        for images, labels in testloader:
            outputs = NeuralNet(images)
            _, predicted = torch.max(outputs.data, 1)
            total += labels.size(0)
            correct += (predicted == labels).sum().item()

    accuracy = correct / total
    print(f'Epoch [{i+1}/{num_epochs}], Validation Accuracy: {100 * accuracy:.2f}%')


