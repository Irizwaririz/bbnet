# This code is based from the following:
# https://scikit-learn.org/stable/auto_examples/linear_model/plot_sparse_logistic_regression_mnist.html
# Author: Arthur Mensch <arthur.mensch@m4x.org>
# License: BSD 3 clause
#
# https://github.com/Fedzbar/deepfedz/blob/master/mnist.py
# Author: Federico Barbero <fb548@cam.ac.uk>
# License: MIT License

import pickle
import numpy as np

from sklearn.metrics import accuracy_score
from sklearn.datasets import fetch_openml
from sklearn.model_selection import train_test_split
from sklearn.utils import check_random_state

from bbnet.nn import NeuralNet
from bbnet.layers import Linear, Tanh
from bbnet.train import train
from bbnet.data import MiniBatchStochastic
from bbnet.optimizers import GD

# Specify number of examples for training and testing
train_samples = 10000
test_samples = 10000

# Load data from https://www.openml.org/d/554
print("Fetching mnist_784 data...")
X, y = fetch_openml('mnist_784', version=1, return_X_y=True)

# Randomize and reshape data before taking training and test examples
print("Randomizing and reshaping data...")
random_state = check_random_state(0)
permutation = random_state.permutation(X.shape[0])
X = X.iloc[permutation].values
y = y.iloc[permutation].values
X = X.reshape((X.shape[0], -1))

# Split data
print("Splitting data...")
X_train, X_test, y_train, y_test = train_test_split(
    X, y, train_size=train_samples, test_size=test_samples)

# Labels come as ['1', '5', ...], we want to convert them to a one hot matrix structure
# Like so: '1' => [0, 1, 0, 0, 0, 0, 0, 0, 0, 0] '5' => [0, 0, 0, 0, 0, 1, 0, 0, 0, 0] and so on
def string_array_to_one_hot_matrix(to_convert: np.ndarray) -> np.ndarray:
    # matrix size: number of labels * 10 options (numbers 0 - 9)
    matrix = np.zeros((to_convert.size, 10))
    matrix[np.arange(to_convert.size), to_convert.astype(int)] = 1
    return matrix

print("Converting labels to one hot matrix...")
y_train = string_array_to_one_hot_matrix(y_train)
y_test = string_array_to_one_hot_matrix(y_test)

print("Initializing neural network...")
net = NeuralNet([
  Linear(input_size=784, output_size=300),
  Tanh(),
  Linear(input_size=300, output_size=100),
  Tanh(),
  Linear(input_size=100, output_size=10),
  Tanh()
])

print("Training neural network...")
train(inputs=X_train, targets=y_train, net=net, iterator=MiniBatchStochastic(batch_size=64), num_epochs=6000, optimizer=GD(lr=0.01))
print("Training complete!")

# Our neural network's predictions come as "probabilities" or "confidence levels" with 
# values ranging from -1 to 1 (since last activation layer is Tanh). An example output is:
# [-0.02299153, 0.01232627, 0.21323621, 0.09431585, -0.02633155, 0.07625834, -0.07023935, -0.08304931, 0.61810449, 0.03938915]
# This means that the prediction of the neural network is 8 since it has the highest value (0.61810449)
# To test our network's accuracy, we need to convert it to like this:
# [0, 0, 0, 0, 0, 0, 0, 0, 1, 0]
# Since this is how we formatted our test labels
def get_max_of_matrix_per_row(matrix: np.ndarray) -> np.ndarray:
    max_matrix = np.zeros_like(matrix)
    max_matrix[np.arange(len(matrix)), matrix.argmax(1)] = 1
    return max_matrix

print("Model Evaluation:")
y_pred = net.forward(X_test)
y_pred = get_max_of_matrix_per_row(y_pred)
accuracy = accuracy_score(y_test, y_pred)
print(accuracy)

print("Saving model...")
pickle.dump(net, open('models/mnist_net.p', 'wb'))
print("Saving complete!")