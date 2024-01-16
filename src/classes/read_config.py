import toml

def read_config(name):
    with open(name, 'r') as file:
        conf = toml.load(file)
    
    return conf.get('settings'), conf.get("layer")