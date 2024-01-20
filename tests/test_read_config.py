import pytest
from src.classes.read_config import read_config


def test_read_config_lr():
    test_config = "config/our_config.toml"

    config_settings, _ = read_config(test_config)

    # Asserting that the learningrate key is present in the config_settings dictionary
    assert 'learningRate' in config_settings

def test_read_config_batchsize():
    test_config = "config/our_config.toml"

    config_settings, _ = read_config(test_config)

    assert 'batchSize' in config_settings

def test_read_config_numepochs():
    test_config = "config/our_config.toml"

    config_settings, _ = read_config(test_config)

    assert 'numEpochs' in config_settings

def test_read_config_type():
    test_config = "config/our_config.toml"

    _, config_layers = read_config(test_config)

    assert 'type' in config_layers[0]

def test_read_config_dims():
    test_config = "config/our_config.toml"

    _, config_layers = read_config(test_config)

    assert 'dims' in config_layers[0]

def test_read_config_activation():
    test_config = "config/our_config.toml"

    _, config_layers = read_config(test_config)

    assert 'activation' in config_layers[0]


if __name__ == '__main__':
    pytest.main()

