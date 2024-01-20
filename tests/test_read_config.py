import pytest
from src.classes.read_config import read_config

def test_read_config_lr():
    test_config = "config/our_config.toml"

    config_settings, _ = read_config(test_config)

    assert 'learningRate' in config_settings

if __name__ == '__main__':
    pytest.main()

