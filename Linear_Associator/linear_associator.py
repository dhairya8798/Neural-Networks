# Parekh, Dhairya
# 1001_868_341
# 2022_10_10
# Assignment_02_01

import numpy as np


class LinearAssociator(object):
    def __init__(self, input_dimensions=2, number_of_nodes=4, transfer_function="Hard_limit"):
        """
        Initialize linear associator model
        :param input_dimensions: The number of dimensions of the input data
        :param number_of_nodes: number of neurons in the model
        :param transfer_function: Transfer function for each neuron. Possible values are:
        "Hard_limit", "Linear".
        """
        self.input_dimensions = input_dimensions
        self.number_of_nodes = number_of_nodes
        self.transfer_function = transfer_function
        self.initialize_weights()

    def initialize_weights(self, seed=None):
        """
        Initialize the weights, initalize using random numbers.
        If seed is given, then this function should
        use the seed to initialize the weights to random numbers.
        :param seed: Random number generator seed.
        :return: None
        """
        nnodes = self.number_of_nodes
        nfeatures = self.input_dimensions

        if seed != None:
            np.random.seed(seed)
            weights = np.random.randn(nnodes, nfeatures)
        else:
            weights = np.random.randn(nnodes, nfeatures)

        self.weights = weights

    def set_weights(self, W):
        """
         This function sets the weight matrix.
         :param W: weight matrix
         :return: None if the input matrix, w, has the correct shape.
         If the weight matrix does not have the correct shape, this function
         should not change the weight matrix and it should return -1.
         """
        nnodes = self.number_of_nodes
        nfeatures = self.input_dimensions
        self.weights = W
        if self.weights.shape != (nnodes, nfeatures):
            return -1
        else:
            return None

    def get_weights(self):
        """
         This function should return the weight matrix(Bias is included in the weight matrix).
         :return: Weight matrix
         """
        weights = self.weights
        return weights

    def predict(self, X):
        """
        Make a prediction on an array of inputs
        :param X: Array of input [input_dimensions,n_samples].
        :return: Array of model outputs [number_of_nodes ,n_samples]. This array is a numerical array.
        """
        weights = self.weights
        nnodes = self.number_of_nodes
        transFunc = self.transfer_function
        netOutput = np.dot(weights, X)
        if transFunc.lower() == "hard_limit":
            netOutput[netOutput >= 0] = 1
            netOutput[netOutput < 0] = 0
            netOutput = np.reshape(netOutput, (nnodes, len(X[0])))
            return netOutput
        else:
            return netOutput

    def fit_pseudo_inverse(self, X, y):
        """
        Given a batch of data, and the targets,
        this function adjusts the weights using pseudo-inverse rule.
        :param X: Array of input [input_dimensions,n_samples]
        :param y: Array of desired (target) outputs [number_of_nodes,n_samples]
        """
        Xpinv = np.linalg.pinv(X)
        self.weights = np.dot(y, Xpinv)

    def train(self, X, y, batch_size=5, num_epochs=10, alpha=0.1, gamma=0.9, learning="Delta"):
        """
        Given a batch of data, and the necessary hyperparameters,
        this function adjusts the weights using the learning rule.
        Training should be repeated num_epochs time.
        :param X: Array of input [input_dimensions,n_samples]
        :param y: Array of desired (target) outputs [number_of_nodes,n_samples].
        :param num_epochs: Number of times training should be repeated over all input data
        :param batch_size: number of samples in a batch
        :param num_epochs: Number of times training should be repeated over all input data
        :param alpha: Learning rate
        :param gamma: Controls the decay
        :param learning: Learning rule. Possible methods are: "Filtered", "Delta", "Unsupervised_hebb"
        :return: None
        """
        weights = self.weights

        for _ in range(num_epochs):
            for j in range(0, X.shape[1], batch_size):
                end_j = j + batch_size
                if end_j > X.shape[1]:
                    end_j = X.shape[1]
                X_slice = X[:, j:end_j]
                y_slice = y[:, j:end_j]
                predictedOutput = self.predict(X_slice)
                if learning.lower() == "delta":
                    self.weights += alpha * (np.dot((y_slice - predictedOutput), X_slice.T))
                elif learning.lower() == "filtered":
                    self.weights = (1-gamma) * weights + alpha * (np.dot(y_slice, X_slice.T))
                else:
                    self.weights += alpha * (np.dot(predictedOutput, X_slice.T))


    def calculate_mean_squared_error(self, X, y):
        """
        Given a batch of data, and the targets,
        this function calculates the mean squared error (MSE).
        :param X: Array of input [input_dimensions,n_samples]
        :param y: Array of desired (target) outputs [number_of_nodes,n_samples]
        :return mean_squared_error
        """
        predictedOutput = self.predict(X)
        mse = np.square(y - predictedOutput).mean()
        return mse