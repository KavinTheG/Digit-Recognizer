from turtle import forward


import numpy as np

class Softmax:

    def __init__(self, X):
        self.X = X

    def forward(self):

        # Get the maximum value of each row
        max_of_each_row = np.max(self.X, axis=1, keepdims=True)

        # Subtract each row by the maximum value in that row
        new_inputs = self.X - max_of_each_row

        # Softmax numerator is the exponeniated value 
        numer = np.exp(new_inputs)

        # Softmax denominator is the sum of the exponeniated value in a row
        denom = np.sum(numer, axis=1, keepdims=True)

        self.result = numer / denom