import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from torchvision import datasets, transforms
from .neural_network_1 import Neural_network

def training_nn(config_layers, lr, batch_size, num_epochs):
    """Load MNIST dataset
       A dataset of handwritten digits
    """
    transform = transforms.Compose([transforms.ToTensor(), transforms.Normalize((0.5,), (0.5,))])

    traindataset = datasets.MNIST(root='./data', train=True, transform=transform, download=True)
    testdataset  = datasets.MNIST(root='./data', train=False, transform=transform, download=True)

    trainloader = DataLoader(traindataset, batch_size=batch_size, shuffle=True)
    testloader  = DataLoader(testdataset, batch_size=batch_size, shuffle=False)

    # Loss function
    criterion = nn.CrossEntropyLoss()

    # Creating an instance of the Neural_network class
    neural_net = Neural_network(config_layers, lr)

    # Training loop
    for i in range(num_epochs):
        for step, (images, labels) in enumerate(trainloader):
            # Forward pass: network processes input images to generate predictions (out)
            out = neural_net(images)
            # loss: compares predicted output with true value (actual labels) 
            loss = criterion(out, labels)
            # backward pass: calculating gradients with respect to the loss for each parameter
            loss.backward()
            # Updates the parameters based on the computed gradients
            neural_net.update()

            # Prints the loss every 100 steps during training to monitor the training process
            if (step + 1) % 100 == 0:
                print(f'Epoch [{i+1}/{num_epochs}], Step [{step+1}/{len(trainloader)}], Loss: {loss.item():.4f}')

        # Calculates and prints the validation accuracy for each epoch 
        correct = 0
        total = 0
        with torch.no_grad():
            for images, labels in testloader:
                outputs = neural_net(images)
                _, predicted = torch.max(outputs.data, 1)
                total += labels.size(0)
                correct += (predicted == labels).sum().item()

        accuracy = correct / total
        print(f'Epoch [{i+1}/{num_epochs}], Validation Accuracy: {100 * accuracy:.2f}%')

