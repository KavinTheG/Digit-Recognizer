import numpy as np
import pandas as pd

from dense_layer import DenseLayer

# Load dataset
csv_file = pd.read_csv("data/train.csv")

# Load labels into numpy array
labels = csv_file['label'].values

# Store rows of pixel values into numpy array
values = csv_file.drop(['label'], axis=1).values

# print(type(values))
# print(csv_file[:5])
# print(labels[:5])#
# print(type(labels))
print(values.shape)

input_layer = DenseLayer(values, 10)
# test = np.dot(values[:1], input_layer.weights[:1].T) # + input_layer.biases[1][1]
# print(values[:1].shape)
# print(input_layer.weights[:1].shape)

print("\n===================X===================\n")
print(pd.DataFrame(values[:1]))
print("\n================Weights================\n")
print(pd.DataFrame(input_layer.weights[:1]))
print("\n================Biases=================\n")
print(pd.DataFrame(input_layer.biases[:1]))


input_layer.forward()
print("\n================Result=================\n")
print(pd.DataFrame(input_layer.result[:1]))
# print(test)