import torch
import pytest
from src.classes.train import training_nn
from src.classes.read_config import read_config
from src.classes.neural_network import Neural_network
from src.classes.load_MNIST import load_MNIST

# Hva skal man egentlig teste på training loopen?
# Hva skal man egentlig teste på neural network?
# Hvorfor funker ikke den ene activation testen vår?

@pytest.fixture
def real_data_loader():
    return load_MNIST

def test_training_nn(real_data_loader):
    # Load configuration from a test file s
    config_settings, config_layers = read_config("config/config_test.toml")

    # Create a real neural network based on config_layers and lr from config_settings
    neural_net = Neural_network(config_layers, config_settings.get('learningRate', 0.001))

    # Call the training function with the real data loader and neural network
    training_nn(config_settings, config_layers, real_data_loader, output_file=None)
