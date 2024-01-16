import toml

def read_config(name):
    with open(name, 'r') as file:
        conf = toml.load(file)
    
    return conf.get('settings'), conf.get("layer")

config_settings, config_layers = read_config('our_input.toml')

lr = config_settings.get('learningRate', 0.0001)
batchSize = config_settings.get('batchSize', 64)
num_epochs = config_settings.get('numEpochs', 10)