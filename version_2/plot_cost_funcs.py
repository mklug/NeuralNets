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
                       
def run_network(filename, num_epochs, eta, lmbda=0.0, cost=network2.CrossEntropyCost):
    """Train the network for ``num_epochs`` with learning
    rate ``eta`` and regularization parameter ``lmbda``.
    Store the resulting training log as a json file at 
    ``filename``.
    """
    # Make results more easily reproducible
    random.seed(12345678)
    np.random.seed(12345678)
    training_data, validation_data, test_data = mnist_loader.load_data_wrapper()
    net = network2.Network([784, 30, 10], cost=cost)
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

    cost_funcs = [network2.CrossEntropyCost, network2.QuadraticCost]
    eta = 0.5
    lmbda = 1.0
    num_epochs = 30

    fig, ax = plt.subplots() 
    ax.set_xlabel('Epoch')
    ax.set_ylabel('Test accuracy')
    ax.set_title('Accuracy on the test data')

    for cost_func in cost_funcs:

        filename = f'training_logs/cf_{str(cost_func)}.json'
        run_network(filename, num_epochs=num_epochs, eta=eta, lmbda=lmbda, cost=cost_func)
        
        f = open(filename, "r")
        test_cost, test_accuracy, training_cost, training_accuracy \
            = json.load(f)
        f.close()
        
        ax.plot(test_accuracy, label=f'cost={str(cost_func)}')

    ax.legend()  
    plt.savefig('compare_various_cost_funcs.png')