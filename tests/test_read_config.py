import pytest
from src.classes.read_config import read_config

@pytest.fixture

def create_file_path():
    return "config/config_test.toml"


def test_read_config_lr():
    test_config = "config/config_test.toml"

    config_settings, _ = read_config(test_config)

    # Asserting that the 'learningRate' value is equal to 0.001
    assert config_settings['learningRate'] == 0.001

def test_read_config_batchsize():
    test_config = "config/config_test.toml"

    config_settings, _ = read_config(test_config)

    assert config_settings['batchSize'] == 64

def test_read_config_numepochs():
    test_config = "config/config_test.toml"

    config_settings, _ = read_config(test_config)

    assert config_settings['numEpochs'] == 5

def test_read_config_type():
    test_config = "config/config_test.toml"

    _, config_layers = read_config(test_config)

    assert config_layers[0]['type'] == 'dense'

def test_read_config_dims():
    test_config = "config/config_test.toml"

    _, config_layers = read_config(test_config)

    assert config_layers[0]['dims'] == [784, 512]

def test_read_config_activation():
    test_config = "config/config_test.toml"

    _, config_layers = read_config(test_config)

    assert config_layers[0]['activation'] == 'relu'


if __name__ == '__main__':
    pytest.main()

    