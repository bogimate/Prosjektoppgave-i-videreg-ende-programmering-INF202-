import argparse
import os
from src.classes.read_config import read_config 
from src.classes.train import training_nn
from src.classes.load_MNIST import load_MNIST

def parse_input():
    parser = argparse.ArgumentParser(description='Neural Network Training Script')

    # Add command-line arguments
    parser.add_argument('-d', '--folder', help='Specify the folder containing config files')
    parser.add_argument('-f', '--file', help='Specify a specific config file')
    parser.add_argument('-o', '--output', help='Specify the output file or folder to store results')
    
    # Parse the command-line arguments
    args = parser.parse_args()

    return args.folder, args.file, args.output


if __name__ == "__main__":
    folder, file, output_file = parse_input()

    if folder:
        # If folder is provided, process all files in the folder
        folder_path = os.path.abspath(folder)
        for filename in os.listdir(folder_path):
            if filename.endswith('.toml'):
                config_path = os.path.join(folder_path, filename)
                print(f"Processing file: {os.path.basename(config_path)}")

                # Read config from the current file in the loop
                config_settings, config_layers = read_config(config_path)

                # Run training and store results
                training_nn(config_settings, config_layers, load_MNIST, output_file=output_file)

    elif file:
        # If a specific file is provided, process only that file
        file_path = os.path.abspath(file)
        print(f"Processing file: {os.path.basename(file_path)}")

        # Read config from the specified file
        config_settings, config_layers = read_config(file_path)

        # Run training and store results
        training_nn(config_settings, config_layers, load_MNIST, output_file=output_file)

    else:
        print("You will now be presented with some options:")
        print("  For running multiple files in a folder, write: python main.py -d (or --folder) path")
        print("  For running one file in a folder, write: python main.py -f (or --file) path/file_name.toml")
        print("  For saving the results, write: python main.py -f (or --file) path/file_name.toml -o (or --output) save_info.txt")
        print()