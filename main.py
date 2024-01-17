import os
import argparse
from src.classes.read_config import read_config 
from src.classes.train import training_nn

# The best working code at the moment.
#    Based on regular U and V (not orthonormal)

# File patch to config file folder
#folder_path = 'config/'

# List of files in the config folder
#folder = os.listdir(folder_path)

# # Loop for reading all the config files in a folder
# for file in folder:
#     if file.endswith('.toml'):
#         # Connects folder path with file name 
#         # Eg: 'config/config_1.toml'
#         config_path = os.path.join(folder_path, file)

#         # Print the current file being processed
#         print(f"Processing file: {config_path}")

#         # Making sure that we read config file from config folder
#         config_settings, config_layers = read_config(config_path)

#         lr = config_settings.get('learningRate', 0.0001)
#         batch_size = config_settings.get('batchSize', 64)
#         num_epochs = config_settings.get('numEpochs', 10)

#         training_nn(config_layers, lr, batch_size, num_epochs)

# def parse_input():
#     parser = argparse.ArgumentParser(description='This is a help message')
    
#     # Add command-line arguments
#     parser.add_argument('-v', '--value', default='default_value', help='Put in a value')
#     parser.add_argument('--flag', action='store_true', help='Set this if flag should be true')
#     parser.add_argument('-d', '--folder', required=True, help='Path to the folder containing files')
#     parser.add_argument('-f', '--file', required=True, help='Name of the specific file to process in the folder')

def parse_input():
    parser = argparse.ArgumentParser(description='Process files in a folder.')
    
    parser.add_argument('-d', '--folder', required=True, help='Path to the folder containing files')
    
    args = parser.parse_args()
    
    folder_path = args.folder
    
    return folder_path

def process_files(folder_path):
    for file in folder_path:
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

if __name__ == "__main__":
    folder_path = parse_input()
    process_files(folder_path)