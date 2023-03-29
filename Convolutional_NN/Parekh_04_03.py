# Parekh, Dhairya
# 1001_868_341
# 2022_11_13
# Assignment_04_03


import numpy as np
import tensorflow.keras as keras
from tensorflow.keras.layers import Conv2D, Dense, Flatten, MaxPooling2D

from cnn import CNN


def test_train():
    
    samples = 300
    n = 10

    (x_train, y_train), (x_test, y_test) = keras.datasets.cifar10.load_data()
    
    x_train = (x_train[0:samples, :]).astype('float64') / 255
    x_test = (x_test[0:samples, :]).astype('float64') / 255

    y_train = keras.utils.to_categorical(y_train[0:samples, :], n)
    y_test = keras.utils.to_categorical(y_test[0:samples, :], n)

    model = keras.Sequential()
    model.add(Conv2D(16, (3, 3), padding = 'same', activation = 'linear', input_shape = x_train.shape[1:]))
    model.add(MaxPooling2D(pool_size = (2, 2)))
    model.add(Conv2D(8, (3, 3), padding = 'same', activation = 'relu'))
    model.add(MaxPooling2D(pool_size = (2, 2)))
    model.add(Flatten())
    model.add(Dense(30, activation = 'relu'))
    model.add(Dense(n, activation = 'softmax'))
    optim = keras.optimizers.RMSprop(learning_rate = 0.001)
    model.compile(optimizer=optim, loss="hinge", metrics=['accuracy'])
    predefined_model = model.fit(x_train, y_train, batch_size = 30, epochs = 50)
    
    my_cnn = CNN()
    my_cnn.add_input_layer(shape=x_train.shape[1:], name="input")
    my_cnn.append_conv2d_layer(num_of_filters=16, kernel_size=(3,3),padding="same", activation='linear', strides = 1,  name="conv1")
    my_cnn.append_maxpooling2d_layer(pool_size=2, padding="same",  strides = 1, name="pool1")
    my_cnn.append_conv2d_layer(num_of_filters=8, kernel_size=3, activation='relu', name="conv2")
    my_cnn.append_maxpooling2d_layer(pool_size=2, padding="same",  strides = 1, name="pool2")
    my_cnn.append_flatten_layer(name="flat1")
    my_cnn.append_dense_layer(num_nodes=30,activation="relu",name="dense1")
    my_cnn.append_dense_layer(num_nodes=n,activation="softmax",name="dense2")
    my_cnn.set_optimizer("rmsprop", 0.001)
    my_cnn.set_metric("accuracy")
    my_cnn.set_loss_function("hinge")
    loss = my_cnn.train(x_train, y_train, batch_size=30, num_epochs=50)

    assert np.allclose(predefined_model.history['loss'], loss, rtol=1e-1, atol=1e-1 * 6)



def test_evaluate():

    samples = 300
    n = 10

    (x_train, y_train), (x_test, y_test) = keras.datasets.cifar10.load_data()
    
    x_train = (x_train[0:samples, :]).astype('float64') / 255
    x_test = (x_test[0:samples, :]).astype('float64') / 255

    y_train = keras.utils.to_categorical(y_train[0:samples, :], n)
    y_test = keras.utils.to_categorical(y_test[0:samples, :], n)

    model = keras.Sequential()
    model.add(Conv2D(16, (3, 3), padding = 'same', activation = 'linear', input_shape = x_train.shape[1:]))
    model.add(MaxPooling2D(pool_size = (2, 2)))
    model.add(Conv2D(8, (3, 3), padding = 'same', activation = 'relu'))
    model.add(MaxPooling2D(pool_size = (2, 2)))
    model.add(Flatten())
    model.add(Dense(30, activation = 'relu'))
    model.add(Dense(n, activation = 'softmax'))
    optim = keras.optimizers.RMSprop(learning_rate = 0.001)
    model.compile(optimizer=optim, loss="hinge", metrics=['accuracy'])
    model.fit(x_train, y_train, batch_size = 30, epochs = 50)
    ls, ac = model.evaluate(x_test, y_test)

    my_cnn = CNN()
    my_cnn.add_input_layer(shape=x_train.shape[1:], name="input")
    my_cnn.append_conv2d_layer(num_of_filters=16, kernel_size=(3,3),padding="same", activation='linear', strides = 1,  name="conv1")
    my_cnn.append_maxpooling2d_layer(pool_size=2, padding="same",  strides = 1, name="pool1")
    my_cnn.append_conv2d_layer(num_of_filters=8, kernel_size=3, activation='relu', name="conv2")
    my_cnn.append_maxpooling2d_layer(pool_size=2, padding="same",  strides = 1, name="pool2")
    my_cnn.append_flatten_layer(name="flat1")
    my_cnn.append_dense_layer(num_nodes=30,activation="relu",name="dense1")
    my_cnn.append_dense_layer(num_nodes=n,activation="softmax",name="dense2")
    my_cnn.set_optimizer("rmsprop", 0.001)
    my_cnn.set_metric("accuracy")
    my_cnn.set_loss_function("hinge")
    my_cnn.train(x_train, y_train, batch_size=30, num_epochs=50)
    loss_value, metric_value = my_cnn.evaluate(x_test, y_test)

    assert np.allclose(ac, metric_value, rtol=1e-1, atol=1e-1 * 6)