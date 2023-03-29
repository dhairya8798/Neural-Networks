# Parekh, Dhairya
# 1001_868_341
# 2022_10_30
# Assignment_03_01

# %tensorflow_version 2.x
import tensorflow as tf
import numpy as np


class MultiNN(object):
    def __init__(self, input_dimension):
        """
        Initialize multi-layer neural network
        :param input_dimension: The number of dimensions for each input data sample
        """
        self.input_dimension = input_dimension
        self.transFuncs = []
        self.weights = []
        self.biases = []

    def add_layer(self, num_nodes, transfer_function="Linear"):
        """
         This function adds a dense layer to the neural network
         :param num_nodes: number of nodes in the layer
         :param transfer_function: Activation function for the layer. Possible values are:
        "Linear", "Relu","Sigmoid".
         :return: None
         """
        if not self.weights:
            currWeights = tf.Variable(np.random.randn(self.input_dimension, num_nodes), trainable=True)
        else:
            currWeights = tf.Variable(np.random.randn(self.weights[-1].shape[1], num_nodes), trainable=True)

        self.weights.append(currWeights)

        currBias = tf.Variable(np.random.randn(num_nodes,), trainable=True)
        self.biases.append(currBias)

        self.transFuncs.append(transfer_function.lower())

    def get_weights_without_biases(self, layer_number):
        """
        This function should return the weight matrix (without biases) for layer layer_number.
        layer numbers start from zero.
         :param layer_number: Layer number starting from layer 0. This means that the first layer with
          activation function is layer zero
         :return: Weight matrix for the given layer (not including the biases). Note that the shape of the weight matrix should be
          [input_dimensions][number of nodes]
         """
        return self.weights[layer_number]

    def get_biases(self, layer_number):
        """
        This function should return the biases for layer layer_number.
        layer numbers start from zero.
        This means that the first layer with activation function is layer zero
         :param layer_number: Layer number starting from layer 0
         :return: Weight matrix for the given layer (not including the biases).
         Note that the biases shape should be [1][number_of_nodes]
         """
        return self.biases[layer_number]

    def set_weights_without_biases(self, weights, layer_number):
        """
        This function sets the weight matrix for layer layer_number.
        layer numbers start from zero.
        This means that the first layer with activation function is layer zero
         :param weights: weight matrix (without biases). Note that the shape of the weight matrix should be
          [input_dimensions][number of nodes]
         :param layer_number: Layer number starting from layer 0
         :return: none
         """
        self.weights[layer_number] = weights

    def set_biases(self, biases, layer_number):
        """
        This function sets the biases for layer layer_number.
        layer numbers start from zero.
        This means that the first layer with activation function is layer zero
        :param biases: biases. Note that the biases shape should be [1][number_of_nodes]
        :param layer_number: Layer number starting from layer 0
        :return: none
        """
        self.biases[layer_number] = biases

    def calculate_loss(self, y, y_hat):
        """
        This function calculates the sparse softmax cross entropy loss.
        :param y: Array of desired (target) outputs [n_samples]. This array includes the indexes of
         the desired (true) class.
        :param y_hat: Array of actual output values [n_samples][number_of_classes].
        :return: loss
        """
        return tf.reduce_mean(tf.nn.sparse_softmax_cross_entropy_with_logits(labels=y, logits=y_hat))


    def predict(self, X):
        """
        Given array of inputs, this function calculates the output of the multi-layer network.
        :param X: Array of input [n_samples,input_dimensions].
        :return: Array of outputs [n_samples,number_of_classes ]. This array is a numerical array.
        """
        tfVar_X = tf.Variable(X)
        for layer in range(len(self.weights)):
            prediction = tf.add(tf.matmul(tfVar_X, self.get_weights_without_biases(layer)), self.get_biases(layer))
            if self.transFuncs[layer] == "linear":
                tfVar_X = prediction
            elif self.transFuncs[layer] == "sigmoid":
                tfVar_X = tf.nn.sigmoid(prediction)
            elif self.transFuncs[layer] == "relu":
                tfVar_X = tf.nn.relu(prediction)
        return tfVar_X

    def train(self, X_train, y_train, batch_size, num_epochs, alpha=0.8):
        """
         Given a batch of data, and the necessary hyperparameters,
         this function trains the neural network by adjusting the weights and biases of all the layers.
         :param X: Array of input [n_samples,input_dimensions]
         :param y: Array of desired (target) outputs [n_samples]. This array includes the indexes of
         the desired (true) class.
         :param batch_size: number of samples in a batch
         :param num_epochs: Number of times training should be repeated over all input data
         :param alpha: Learning rate
         :return: None
         """
        for _ in range(num_epochs):
            for j in range(0, X_train.shape[0], batch_size):
                end_j = j + batch_size
                if end_j > X_train.shape[0]:
                    end_j = X_train.shape[0]
                X_slice = tf.Variable(X_train[j:end_j, :])
                y_slice = tf.Variable(y_train[j:end_j])

                with tf.GradientTape(persistent=True) as tape:
                    tape.watch(self.weights)
                    tape.watch(self.biases)
                    y_hat = self.predict(X_slice)
                    loss = self.calculate_loss(y_slice, y_hat)

                for layer in range(len(self.weights)):
                    layer_w = tf.scalar_mul(alpha, tape.gradient(loss, self.get_weights_without_biases(layer)))
                    layer_b = tf.scalar_mul(alpha, tape.gradient(loss, self.get_biases(layer)))
                    weights = tf.subtract(self.get_weights_without_biases(layer), layer_w)
                    bias = tf.subtract(self.get_biases(layer), layer_b)
                    self.set_weights_without_biases(weights, layer)
                    self.set_biases(bias, layer)
        

    def calculate_percent_error(self, X, y):
        """
        Given input samples and corresponding desired (true) output as indexes,
        this method calculates the percent error.
        For each input sample, if the predicted class output is not the same as the desired class,
        then it is considered one error. Percent error is number_of_errors/ number_of_samples.
        Note that the predicted class is the index of the node with maximum output.
        :param X: Array of input [n_samples,input_dimensions]
        :param y: Array of desired (target) outputs [n_samples]. This array includes the indexes of
        the desired (true) class.
        :return percent_error
        """
        prediction = self.predict(X).numpy()
        yPred = np.argmax(prediction, axis=1)
        error = 0
        for i in range(prediction.shape[0]):
            if yPred[i] != y[i]:
                error += 1
        percentError = error/prediction.shape[0]
        return percentError

    def calculate_confusion_matrix(self, X, y):
        """
        Given input samples and corresponding desired (true) outputs as indexes,
        this method calculates the confusion matrix.
        :param X: Array of input [n_samples,input_dimensions]
        :param y: Array of desired (target) outputs [n_samples]. This array includes the indexes of
        the desired (true) class.
        :return confusion_matrix[number_of_classes,number_of_classes].
        Confusion matrix should be shown as the number of times that
        an image of class n is classified as class m.
        """
        prediction = self.predict(X).numpy()
        yPred = np.argmax(prediction, axis=1)
        cnt = len(np.unique(y))
        cm = np.zeros((cnt, cnt))
        for i in range(len(y)):
            r = y[i].astype(int)
            c = yPred[i].astype(int)
            cm[r][c] = cm[r][c] + 1
        print(cm)
        return cm
