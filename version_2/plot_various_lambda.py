"""
Plot graphs to illustrate changing the learning rate.  
"""
# Standard library
import json
import random
import sys

# My library
import network2
sys.path.append('../')
import mnist_loader

# Third-party libraries
import matplotlib.pyplot as plt
import numpy as np
                       
def run_network(filename, num_epochs, eta, lmbda=0.0):
    """Train the network for ``num_epochs`` with learning
    rate ``eta`` and regularization parameter ``lmbda``.
    Store the resulting training log as a json file at 
    ``filename``.
    """
    # Make results more easily reproducible
    random.seed(12345678)
    np.random.seed(12345678)
    training_data, validation_data, test_data = mnist_loader.load_data_wrapper()
    net = network2.Network([784, 30, 10], cost=network2.QuadraticCost())
    net.large_weight_initializer()
    test_cost, test_accuracy, training_cost, training_accuracy \
        = net.SGD(training_data, num_epochs, 10, eta,
                  evaluation_data=test_data, lmbda = lmbda,
                  monitor_evaluation_cost=True, 
                  monitor_evaluation_accuracy=True, 
                  monitor_training_cost=True, 
                  monitor_training_accuracy=True)
    f = open(filename, "w")
    json.dump([test_cost, test_accuracy, training_cost, training_accuracy], f)
    f.close()

if __name__ == "__main__":

    lmbdas = [0.001, 0.01, 0.1, 0.0, 1.0, 10.0, 100.0]
    fig, ax = plt.subplots() 
    ax.set_xlabel('Epoch')
    ax.set_ylabel('Test accuracy')
    ax.set_title('Accuracy on the test data')

    for lmbda in lmbdas:

        filename = f'training_logs/lmbda_{lmbda}.json'
        run_network(filename, num_epochs=30, eta=0.5, lmbda=lmbda)
        
        f = open(filename, "r")
        test_cost, test_accuracy, training_cost, training_accuracy \
            = json.load(f)
        f.close()
        
        ax.plot(test_accuracy, label=f'lmbda={lmbda}')

    ax.legend()  
    plt.savefig('compare_various_lmbdas.png')