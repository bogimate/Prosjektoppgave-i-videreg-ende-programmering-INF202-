import os
from src.classes.read_config import read_config 
from classes.train_2 import training_nn

# The best working code at the moment.
#    Based on regular U and V (not orthonormal)

# File patch to config file folder
folder_path = 'config/'

# List of files in the config folder
folder = os.listdir(folder_path)


# Loop for reading all the config files in a folder
for file in folder:
    if file.endswith('.toml'):
        # Connects folder path with file name 
        # Eg: 'config/config_1.toml'
        config_path = os.path.join(folder_path, file)

        # Print the current file being processed
        print(f"Processing file: {config_path}")

        # Making sure that we read config file from config folder
        config_settings, config_layers = read_config(config_path)

        lr = config_settings.get('learningRate', 0.0001)
        batch_size = config_settings.get('batchSize', 64)
        num_epochs = config_settings.get('numEpochs', 10)

        training_nn(config_layers, lr, batch_size, num_epochs)

