import torch
import torch.nn as nn
from .neural_network import Neural_network


def training_nn(config_settings, config_layers, data_loader_func, output_file='training_info.txt'):
    """
    Trains a neural network based on input arguments.
    Args:
        config_setting: Setting parameters from a config file
        config_layers: Layer parameters from a config file
        data_loader_func: A function for loading data
        output_file: Output file name for storing loss and accuracy
    """

    criterion = nn.CrossEntropyLoss()

    # Retriving the settings from config_settings
    lr = config_settings.get('learningRate', 0.0001)
    batch_size = config_settings.get('batchSize', 64)
    num_epochs = config_settings.get('numEpochs', 10)

    # Defining train loader and test loader from a specified function
    trainloader, testloader = data_loader_func(batch_size)

    # Making a neural network based on config_layers and lr from config_settings
    neural_net = Neural_network(config_layers, lr)

    if output_file:
        # If output file is specified, redirect the print to the specified file
        with open(output_file, 'a') as info_file:
            # Training loop
            for i in range(num_epochs):
                # Iterates over the batches in the trainloader. step is the index of the current batch.
                for step, (images, labels) in enumerate(trainloader):
                    # Forward pass: network processes input images to generate predictions (out)
                    out = neural_net(images)
                    # loss: compares predicted output with true value (labels) 
                    loss = criterion(out, labels)
                    # backward pass: calculating gradients with respect to the loss for each parameter
                    loss.backward()
                    # Updates the parameters based on the computed gradients
                    neural_net.update()

                    # Prints the loss every 100 steps during training to monitor the training process
                    if (step + 1) % 100 == 0:
                        print(f'Epoch [{i+1}/{num_epochs}], Step [{step+1}/{len(trainloader)}], Loss: {loss.item():.4f}', file=info_file)


                # Calculates and prints the validation accuracy for each epoch 
                correct = 0  # Counter for correctly predicted labels
                total = 0    # Counter for total number of labels
                with torch.no_grad(): # Ensure that no gradients are computed during the validation phase
                    # Iterates over the batches
                    for images, labels in testloader:  
                        # Model's predictions 
                        outputs = neural_net(images)
                        # Finds the index of the maximum value in outputs.data
                        _, predicted = torch.max(outputs.data, 1)
                        # Total number of label
                        total += labels.size(0)
                        # Summing up the correct predictions
                        correct += (predicted == labels).sum().item()
                
                # Calculate the accuracy
                accuracy = correct / total
                
                # Prints and write the information to the output_file
                print(f'Epoch [{i+1}/{num_epochs}], Validation Accuracy: {100 * accuracy:.2f}%', file=info_file)
                
                # saving the values
                neural_net.save(f'model_parameters_epoch_{i + 1}.pth')

    else:
        # If no output file is specified, print the results to the terminal
        for i in range(num_epochs):
            for step, (images, labels) in enumerate(trainloader):
                out = neural_net(images)
                loss = criterion(out, labels)
                loss.backward()
                neural_net.update()

                if (step + 1) % 100 == 0:
                    print(f'Epoch [{i + 1}/{num_epochs}], Step [{step + 1}/{len(trainloader)}], Loss: {loss.item():.4f}')

            correct = 0
            total = 0
            with torch.no_grad():
                for images, labels in testloader:
                    outputs = neural_net(images)
                    _, predicted = torch.max(outputs.data, 1)
                    total += labels.size(0)
                    correct += (predicted == labels).sum().item()

            accuracy = correct / total
            print(f'Epoch [{i + 1}/{num_epochs}], Validation Accuracy: {100 * accuracy:.2f}%')
            