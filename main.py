import os
from src.classes.read_config import read_config 
from src.classes.train import training_nn

# Final Validation Accuracy 88.83%
# The best working code at the moment without orthonormal U and V
# With config read function

# Making sure that we read config file from config folder
# config_settings, config_layers = read_config('config/our_config.toml')

# lr = config_settings.get('learningRate', 0.0001)
# batch_size = config_settings.get('batchSize', 64)
# num_epochs = config_settings.get('numEpochs', 10)

# training_nn(config_layers, lr, batch_size, num_epochs)

config_folder_path = 'config/'

folder = os.listdir(config_folder_path)

for file in folder:
    if file.endswith('.toml'):
        config_path = os.path.join(config_folder_path, file)

        # Making sure that we read config file from config folder
        config_settings, config_layers = read_config('config/our_config.toml')

        lr = config_settings.get('learningRate', 0.0001)
        batch_size = config_settings.get('batchSize', 64)
        num_epochs = config_settings.get('numEpochs', 10)

        training_nn(config_layers, lr, batch_size, num_epochs)

