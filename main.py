import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from torchvision import datasets, transforms
import toml
from src.classes.neural_network import Neural_network
from src.classes.read_config import read_config 
from src.classes.train import training_nn

# Final Validation Accuracy 88.83%
# The best working code at the moment without orthonormal U and V
# With config read function

# Makeing sure that we read config file from config folder
config_settings, config_layers = read_config('config/our_config.toml')

lr = config_settings.get('learningRate', 0.0001)
batch_size = config_settings.get('batchSize', 64)
num_epochs = config_settings.get('numEpochs', 10)

training_nn(config_layers, lr, batch_size, num_epochs)

# # Load MNIST dataset
# transform = transforms.Compose([transforms.ToTensor(), transforms.Normalize((0.5,), (0.5,))])

# traindataset = datasets.MNIST(root='./data', train=True, transform=transform, download=True)
# testdataset  = datasets.MNIST(root='./data', train=False, transform=transform, download=True)

# trainloader = DataLoader(traindataset, batch_size=batchSize, shuffle=True)
# testloader  = DataLoader(testdataset, batch_size=batchSize, shuffle=False)


# criterion = nn.CrossEntropyLoss()
# # Create an instance of Neural_network and directly use the activation functions from the Activation class
# NeuralNet = Neural_network(config_layers, lr)

# # Training loop
# for i in range(num_epochs):
#     for step, (images, labels) in enumerate(trainloader):
#         # Forward pass: network processes input images to generate predictions (out)
#         out = NeuralNet(images)
#         # loss = compares predicted output with true value (actual labels) 
#         loss = criterion(out, labels)
#         # backward pass: calculating gradients with respect to the loss for each parameter
#         loss.backward()
#         # Updates the parameters based on the computed gradients
#         NeuralNet.update()

#         # TODO: Train your weights and biases. Think about how you can use object oriented programming to do so.
#         # The way you do this heavily impacts the extendability of your code. So take some time to think about the design of your program!

#         # Prints the loss every 100 steps during training to monitor the training process
#         if (step + 1) % 100 == 0:
#             print(f'Epoch [{i+1}/{num_epochs}], Step [{step+1}/{len(trainloader)}], Loss: {loss.item():.4f}')

#     # Calculates and prints the validation accuracy for each epoch 
#     correct = 0
#     total = 0
#     with torch.no_grad():
#         for images, labels in testloader:
#             outputs = NeuralNet(images)
#             _, predicted = torch.max(outputs.data, 1)
#             total += labels.size(0)
#             correct += (predicted == labels).sum().item()

#     accuracy = correct / total
#     print(f'Epoch [{i+1}/{num_epochs}], Validation Accuracy: {100 * accuracy:.2f}%')


