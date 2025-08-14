# INF202 Low-rank neural network
# USER GUIDE
To run this software, start by making a new environment and install the necessary packages from the requirements file. This can be done by writing the following in the terminal:
    pip install -r requirements.txt

Then go to main.py and run the code from there. A default config file will be processed. After the processing, the option to either run a folder of config files or just one config file will be presented. 

How to run a folder of config files:
    python main.py -d filepath
    python main.py –-folder filepath

How to run a specific config file:
    python main.py -f filepath/config_name.toml
    python main.py –-file filepath/config_name.tom

The loss and accuracy will be printed in the terminal in real time. Another possibility is
to save the loss, accuracy and the models parameters instead of getting it printed.
How to save loss and accuracy for one file:
    python main.py –f filepath/config_name.toml -o save_name.txt
    python main.py –-file filepath/config_name.toml --output save_results.txt

# HOW TO:
# Add activation functions
Create an instence of the Activation factory:
    activation_factory = Activation_factory()
Adding the new activation function:
    activation_factory.register('new_activation_function', nn.New_activation_function())
Note that this statement takes a key and a name, meaning string and PyTorch function from PyTorch.nn. 
If it already exists, the code will raise a ValueError.
