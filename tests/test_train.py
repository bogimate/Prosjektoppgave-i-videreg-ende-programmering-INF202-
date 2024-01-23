import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from src.classes.neural_network import Neural_network
from src.classes.train import training_nn



def test_training_nn():
    # Mock configuration settings
    config_settings = {
        'learningRate': 0.001,
        'batchSize': 32,
        'numEpochs': 2,
    }

    # Mock configuration layers
    config_layers = [
        {'type': 'dense', 'dims': [784, 256], 'activation': 'relu'},
        {'type': 'dense', 'dims': [256, 10], 'activation': 'softmax'},
    ]

    # Mock data loader function
    def data_loader_func(batch_size):
        # Return mock train and test loaders
        train = DataLoader(torch.randn(1000, 784), batch_size=batch_size)
        test = DataLoader(torch.randn(100, 784), batch_size=batch_size)
        return train, test

    # Run training
    training_nn(config_settings, config_layers, data_loader_func)