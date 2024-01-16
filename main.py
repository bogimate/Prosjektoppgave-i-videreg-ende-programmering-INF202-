from src.classes.neural_network import Neural_Net
import toml
import argparse

def parse_input():
    '''Parse input arguments to determine config name'''
    parser = argparse.ArgumentParser(description = 'Input config file name with -f or --file')
    parser.add_argument('-f', '--file', default='config.toml', help='Put in name of config file.')
    args = parser.parse_args()
    return args.file

def read_config(name):
    with open(name, 'r') as file:
        conf = toml.load(file)

    return conf.get("layer")

# if __name__ == "__main__":
#     conf_name = parse_input()
#     players = read_config(conf_name)

