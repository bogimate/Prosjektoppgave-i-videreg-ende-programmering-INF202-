import toml


def read_config(name):
    """
    Returns the settings and the layers from a config file. 
    Args:
        name: name of the config file
    Returns:
        output settings and layers for given input
    """
    with open(name, 'r') as file:
        conf = toml.load(file)
    
    return conf.get('settings'), conf.get("layer")
