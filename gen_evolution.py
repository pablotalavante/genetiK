'''Trains a simple deep NN on the MNIST dataset.
Gets to 98.40% test accuracy after 20 epochs
(there is *a lot* of margin for parameter tuning).
2 seconds per epoch on a K520 GPU.
'''

from __future__ import print_function

import keras
from keras.datasets import mnist
from keras.models import Sequential
from keras.layers import Dense, Dropout
from keras.optimizers import RMSprop

batch_size = 128
num_classes = 10
epochs = 2

# the data, split between train and test sets
(x_train, y_train), (x_test, y_test) = mnist.load_data()

x_train = x_train[:2000].reshape(2000, 784)
x_test = x_test[2000:3000].reshape(1000, 784)
x_train = x_train.astype('float32')
x_test = x_test.astype('float32')
x_train /= 255
x_test /= 255
print(x_train.shape[0], 'train samples')
print(x_test.shape[0], 'test samples')

# convert class vectors to binary class matrices
y_train = keras.utils.to_categorical(y_train[:2000], num_classes)
y_test = keras.utils.to_categorical(y_test[2000:3000], num_classes)


def network(x_train, y_train, x_test, y_test, d_1, d_2):
    model = Sequential()
    model.add(Dense(d_1, activation='relu', input_shape=(784,)))
    model.add(Dense(d_2, activation='relu'))
    model.add(Dense(num_classes, activation='softmax'))

    model.summary()

    model.compile(loss='categorical_crossentropy',
                  optimizer=RMSprop(),
                  metrics=['accuracy'])

    history = model.fit(x_train, y_train,
                        batch_size=batch_size,
                        epochs=epochs,
                        verbose=1,
                        validation_data=(x_test, y_test))
    score = model.evaluate(x_test, y_test, verbose=0)
    print('Test loss:', score[0])
    print('Test accuracy:', score[1])
    
    return float(score[0])    #this float is important




from nevergrad import instrumentation as instru

def myfunction(arg1, arg2, arg3, value=3):
    print(arg1, arg2, arg3)
    return value**2

# argument transformation
first_dense_neurons = instru.variables.OrderedDiscrete([10, 20])  # neurons of the first dense layer
second_dense_neurons = instru.variables.OrderedDiscrete([10, 30])  # 1st arg. = positional discrete argument
dropout_rate = instru.variables.OrderedDiscrete([0, 0.2, 0.6])  # 2nd arg. = positional discrete argument

# create the instrumented function
ifunc = instru.InstrumentedFunction(network, x_train, y_train, x_test, y_test,
                                    first_dense_neurons, second_dense_neurons)
# the 3rd arg. is a positional arg. which will be kept constant to "blublu"
print('La dimensión del problema es: ', ifunc.dimension)  # 5 dimensional space

from nevergrad.optimization import optimizerlib

optimizer = optimizerlib.CMA(dimension=ifunc.dimension, budget=5, num_workers=1)

for _ in range(optimizer.budget):
    print('epoch: ', _)
    x = optimizer.ask()
    print('X: ', x)
    value = ifunc(x)
    print('value: ', value)
    optimizer.tell(x, value)
    print('optimizer: ', optimizer)

recommendation = optimizer.provide_recommendation()

print(recommendation)

ifunc(recommendation)



