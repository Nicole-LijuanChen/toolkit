import numpy as np
from tensorflow import keras

from tensorflow import keras
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, Dropout, Activation
from tensorflow.keras.optimizers import SGD
from tensorflow.keras.datasets import mnist
from tensorflow.keras.utils import to_categorical




def define_nn_mlp_model(X_train, y_train_ohe):
    ''' defines multi-layer-perceptron neural network '''
    # available activation functions at:
    # https://keras.io/activations/
    # https://en.wikipedia.org/wiki/Activation_function
    # options: 'linear', 'sigmoid', 'tanh', 'relu', 'softplus', 'softsign'
    # there are other ways to initialize the weights besides 'uniform', too 
    
    model = Sequential() # sequence of layers
    num_neurons_in_layer = 100 # number of neurons in a layer
    num_inputs = X_train.shape[1] # number of features (784)
    num_classes = y_train_ohe.shape[1]  # number of classes, 0-9
    model.add(Dense(units=num_neurons_in_layer, 
                    input_shape=(num_inputs, ),
                    kernel_initializer='uniform', 
                    bias_initializer='zeros',
                    activation='relu')) 
    model.add(Dense(units=num_neurons_in_layer, 
                    kernel_initializer='uniform', 
                    bias_initializer='zeros',
                    activation='relu')) 
    model.add(Dense(units=num_classes,
                    kernel_initializer='uniform', 
                    bias_initializer='zeros',
                    activation='softmax')) 
    sgd = SGD(lr=0.001, decay=1e-7, momentum=.9) # using stochastic gradient descent 
    model.compile(loss='categorical_crossentropy', optimizer=sgd, metrics=["accuracy"] ) 
    return model 