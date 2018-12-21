#!/usr/bin/env python
# coding: utf-8

# In[42]:


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


# In[68]:


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


# In[69]:


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
print(ifunc.dimension)  # 5 dimensional space

# The dimension is 5 because:
# - the 1st discrete variable has 1 possible values, represented by a hard thresholding in a 1-dimensional space, i.e. we add 1 coordinate to the continuous problem
# - the 2nd discrete variable has 3 possible values, represented by softmax, i.e. we add 3 coordinates to the continuous problem
# - the 3rd variable has no uncertainty, so it does not introduce any coordinate in the continuous problem
# - the 4th variable is a real number, represented by single coordinate.

ifunc([1, -80])  # will print "b e blublu" and return 49 = (mean + std * arg)**2 = (1 + 2 * 3)**2
# b is selected because 1 > 0 (the threshold is 0 here since there are 2 values.
# e is selected because proba(e) = exp(80) / (exp(80) + exp(-80) + exp(-80))






# In[70]:


'''
not ask and tell


from nevergrad.optimization import optimizerlib

optimizer = optimizerlib.CMA(dimension=2, budget=5, num_workers=1)

from concurrent import futures

with futures.ThreadPoolExecutor(max_workers=optimizer.num_workers) as executor:
    recommendation = optimizer.optimize(ifunc, executor=executor, batch_mode=True)
'''


# In[71]:


for _ in range(optimizer.budget):
    print('epoch: ', _)
    x = optimizer.ask()
    print('X: ', x)
    value = ifunc(x)
    print('value: ', value)
    optimizer.tell(x, value)
    print('optimizer: ', optimizer)

recommendation = optimizer.provide_recommendation()


# In[72]:


type(recommendation)


# In[73]:


print(recommendation)


# In[74]:


ifunc(recommendation)


# In[ ]:




