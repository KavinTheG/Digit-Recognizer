from audioop import bias
import numpy as np

class DenseLayer():

    # output_neuron_nums stores the number of neurons
    # in the next layer
    def __init__(self, X, output_neuron_nums):
        self.X = X

        # This layer will have outputs for 10 neurons
        # There, must create 10 sets of weights 'layers' 
        # Each set has 724 weight
        self.weights = np.random.rand(output_neuron_nums, X.shape[1]) / 1000

        # Number of biases will match the number of output nerons
        self.biases = np.random.rand(1, output_neuron_nums) / 1000


    def forward(self):
        self.result = np.matmul(self.X, self.weights.T) + self.biases

        self.result[self.result < 0] = 0