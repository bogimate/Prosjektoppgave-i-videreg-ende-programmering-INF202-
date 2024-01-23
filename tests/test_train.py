import torch
import pytest
from src.classes.train import training_nn
from src.classes.read_config import read_config
from src.classes.neural_network import Neural_network
from src.classes.load_MNIST import load_MNIST


@pytest.fixture
def real_data_loader():
    return load_MNIST

def test_training_nn(real_data_loader):
    # Load configuration from a test file s
    config_settings, config_layers = read_config("config/config_test.toml")