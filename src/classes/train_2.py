import torch
import torch.nn as nn
from .neural_network import Neural_network
# from torch.utils.data import DataLoader
# from torchvision import datasets, transforms



def training_nn(config_settings, config_layers, data_loader_func):
    criterion = nn.CrossEntropyLoss()

    # Retriving the settings from config_settings
    lr = config_settings.get('learningRate', 0.0001)
    batch_size = config_settings.get('batchSize', 64)
    num_epochs = config_settings.get('numEpochs', 10)

    # Defining train loader and test loader from a specyfied function
    trainloader, testloader = data_loader_func(batch_size)

    # Makeing a neural network based on config_layers and lr from config_settings
    neural_net = Neural_network(config_layers, lr)

    # traing loop
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

