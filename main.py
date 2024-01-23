import os
import argparse
import torch.nn as nn
from src.classes.read_config import read_config 
from classes.train_2 import training_nn
from src.classes.load_MNIST import load_MNIST

def parse_input():
    parser = argparse.ArgumentParser(description='Neural Network Training Script')

    # Add command-line arguments
    parser.add_argument('-d', '--folder', help='Specify the folder containing config files')
    parser.add_argument('-f', '--file', help='Specify a specific config file')

    # Parse the command-line arguments
    args = parser.parse_args()

    return args.folder, args.file 

if __name__ == "__main__":
    folder, file = parse_input()

    if folder:
        # If folder is provided, process all files in the folder
        folder_path = os.path.abspath(folder)
        for filename in os.listdir(folder_path):
            if filename.endswith('.toml'):
                config_path = os.path.join(folder_path, filename)
                print(f"Processing file: {os.path.basename(config_path)}")

                # Read config from the current file in the loop
                config_settings, config_layers = read_config(config_path)

                #lr = config_settings.get('learningRate', 0.0001)
                #batch_size = config_settings.get('batchSize', 64)
                #num_epochs = config_settings.get('numEpochs', 10)

                # training_nn(config_layers, lr, batch_size, num_epochs)
                training_nn(config_settings, config_layers, load_MNIST)

    elif file:
        # If a specific file is provided, process only that file
        file_path = os.path.abspath(file)
        print(f"Processing file: {os.path.basename(file_path)}")

        # Read config from the specified file
        config_settings, config_layers = read_config(file_path)

        #lr = config_settings.get('learningRate', 0.0001)
        #batch_size = config_settings.get('batchSize', 64)
        #num_epochs = config_settings.get('numEpochs', 10)

        training_nn(config_settings, config_layers, load_MNIST)

        # training_nn(config_layers, lr, batch_size, num_epochs)
        #training_nn(neural_net, trainloader, testloader, num_epochs=10)

    else:
        print("Please provide either -d (or --folder) or -d (or --folder) and -f (or --file) argument.")
        print("E.g: python file_name.py -d path_name")
        print(r'''E.g: python file_name.py -f path_name\file_name.toml''')
        print()


# # Loss function
# criterion = nn.CrossEntropyLoss()

# # Creating an instance of the Neural_network class
#neural_net = Neural_network(config_layers, lr)

