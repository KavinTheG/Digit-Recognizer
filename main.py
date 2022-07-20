import numpy as np
import pandas as pd

from dense_layer import DenseLayer
from loss_categoricalentropy import LossCategoricalEntropy
from softmax import Softmax

# Load dataset
csv_file = pd.read_csv("data/train.csv") 

# Load labels into numpy array
labels = csv_file['label'].values

# Store rows of pixel values into numpy array
values = csv_file.drop(['label'], axis=1).values

print(values.shape)

# print(csv_file[:5])
print("\nLabels: ")
print(labels[:5])#

labels_max = np.max(labels) + 1
y_target = np.eye(labels_max)[labels]

print("\nOne-Hot Encoded: ")
print(y_target[:5])#
print(y_target.shape)

input_layer = DenseLayer(values, 10)

print("\n===================X===================\n")
print(pd.DataFrame(values[:1]))
print("\n================Weights================\n")
print(pd.DataFrame(input_layer.weights[:1]))
print("\n================Biases=================\n")
print(pd.DataFrame(input_layer.biases[:1]))


input_layer.forward()
print("\n================Result=================\n")
print(pd.DataFrame(input_layer.result[:1]))

# Second layer
hidden_layer = DenseLayer(input_layer.result, 10)
hidden_layer.forward()

# Final layer - Softmax Activation
output_layer = Softmax(hidden_layer.result)
output_layer.forward()

print("\nSoftmax Output")
print(pd.DataFrame(output_layer.result[:5]))
print(output_layer.result.shape)

# Loss Function
loss_function = LossCategoricalEntropy(output_layer.result, y_target)
loss_function.calculate_loss()

print("\nLoss Result")
#print(loss_function.result[:5])

