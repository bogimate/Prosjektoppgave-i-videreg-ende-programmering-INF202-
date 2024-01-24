from src.classes.load_MNIST import load_MNIST

def test_load_MNIST_trainloader():
    batch_size = 64
    trainloader, _ = load_MNIST(batch_size)

    assert trainloader is not None

def test_load_MNIST_testloader():
    batch_size = 64
    _, testloader = load_MNIST(batch_size)

    assert testloader is not None
