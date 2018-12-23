#!/usr/bin/env python
# coding: utf-8

# In[219]:


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

import matplotlib.pyplot as plt

batch_size = 128
num_classes = 10
epochs = 10

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


# In[220]:


import numpy as np


# In[221]:


from numpy.random import seed
seed(1)


# In[222]:


def network(x_train, y_train, x_test, y_test, d_1, d_2, dr, if_dense3, d_3, act):
    model = '_'
    model = Sequential()
    model.add(Dense(d_1, activation=act, input_shape=(784,)))
    model.add(Dropout(dr))
    model.add(Dense(d_2, activation=act))
    if if_dense3:
        model.add(Dense(d_3, activation=act))
    model.add(Dense(num_classes, activation='softmax'))

    #model.summary()

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


# In[223]:


from nevergrad import instrumentation as instru

# argument transformation
first_dense_neurons = instru.variables.OrderedDiscrete([10, 20])  # neurons of the first dense layer
second_dense_neurons = instru.variables.OrderedDiscrete([10, 30])  # 1st arg. = positional discrete argument
dropout_rate = instru.variables.OrderedDiscrete([0.2, 0.6])  # 2nd arg. = positional discrete argument

third_dense = instru.variables.SoftmaxCategorical([False, True]) #Wether to add a third dense layer
third_dense_neurons = instru.variables.OrderedDiscrete([10, 50, 80]) #And its number of neurons if True

activation = instru.variables.SoftmaxCategorical(['relu', 'elu']) # Let's also play with that

# create the instrumented function
ifunc = instru.InstrumentedFunction(network, x_train, y_train, x_test, y_test,
                                    first_dense_neurons, second_dense_neurons, dropout_rate,
                                    third_dense, third_dense_neurons,
                                    activation)

# the 3rd arg. is a positional arg. which will be kept constant to "blublu"
print('Dimensiones: ', ifunc.dimension)  # 5 dimensional space

# The dimension is 5 because:
# - the 1st discrete variable has 1 possible values, represented by a hard thresholding in a 1-dimensional space, i.e. we add 1 coordinate to the continuous problem
# - the 2nd discrete variable has 3 possible values, represented by softmax, i.e. we add 3 coordinates to the continuous problem
# - the 3rd variable has no uncertainty, so it does not introduce any coordinate in the continuous problem
# - the 4th variable is a real number, represented by single coordinate.

#ifunc([1, -80])  # will print "b e blublu" and return 49 = (mean + std * arg)**2 = (1 + 2 * 3)**2
# b is selected because 1 > 0 (the threshold is 0 here since there are 2 values.
# e is selected because proba(e) = exp(80) / (exp(80) + exp(-80) + exp(-80))


# In[207]:


get_ipython().run_line_magic('pinfo2', 'instru.variables')


# In[244]:


from nevergrad.optimization import optimizerlib

optimizer = optimizerlib.CMA(dimension=ifunc.dimension, budget=200, num_workers=1)


# In[216]:


from concurrent import futures

with futures.ThreadPoolExecutor(max_workers=optimizer.num_workers) as executor:
    recommendation = optimizer.optimize(ifunc, executor=executor, batch_mode=True, verbosity=2)


# In[245]:


history = []

for _ in range(optimizer.budget):
        print('\n\nGeneration: ', _)
        x = [0]*optimizer.num_workers
        for worker in range(optimizer.num_workers):
            x[worker] = optimizer.ask()
            print('X: ', x)
            value = ifunc(x[worker])
        history.append([_, x[worker], value])
        print('value: ', value)
        optimizer.tell(x[worker], value)
        print('optimizer: ', optimizer)

recommendation = optimizer.provide_recommendation()


# In[246]:


history


# In[323]:


x_scatter = []
y_scatter = []
s_scatter = []
ep_scatter = []
for h in history:
    l = []
    for p in range(len(h[1])):
        l.append(h[1][p])
    x_scatter.append(l)
    y_scatter.append(h[1][5])
    s_scatter.append(0.3*(1/h[2]**10))
    ep_scatter.append(0.4*h[0]/len(history))


# In[314]:


x_scatter


# In[321]:


plt.figure(figsize=(10,8))
for i in range(len(ep_scatter)):
    plt.scatter(x_scatter[i][0], x_scatter[i][1], color='r', s=s_scatter[i], alpha = ep_scatter[i])
    if i < len(ep_scatter)-1:
        plt.plot([x_scatter[i-1][0]] + [x_scatter[i][1]], [x_scatter[i-1][0]] + [x_scatter[i][1]], color='black', alpha=0.4*i/len(ep_scatter))

plt.show()


# In[ ]:


f = plt.figure()    
nrows = 8
ncols = 8
f, axes = plt.subplots(nrows = nrows, ncols = ncols, sharex=True, sharey = True, figsize=(50,50))

for col in range(ncols):
    for row in range(nrows):
        for i in range(len(ep_scatter)):
            axes[row][col].scatter(x_scatter[i][col], x_scatter[i][row],
                                   color='r', s=s_scatter[i], alpha = ep_scatter[i])
            if i < len(ep_scatter)-1:
                axes[row][col].plot([x_scatter[i-1][col]] + [x_scatter[i][col]],
                                    [x_scatter[i-1][row]] + [x_scatter[i][row]],
                                    color='black', alpha=0.4*i/len(ep_scatter))

plt.savefig('params_.png', dpi=100)      
plt.show()


# In[72]:


type(recommendation)


# In[73]:


print(recommendation)


# In[74]:


ifunc(recommendation)


# In[272]:


args, kwargs = ifunc.convert_to_arguments(recommendation)
print(args)    # should print ["b", "e", "blublu"]
print(kwargs)  # should print {"value": 7}

# but be careful, since some variables are stochastic (SoftmaxCategorical ones are), so several runs may lead to several results
# The following will print more information on the conversion to your arguments:
print(ifunc.get_summary(recommendation))


# In[201]:


args[0].shape


# In[206]:


plt.imshow(np.reshape(args[0][0], (28, 28)))


# In[194]:


print(ifunc.get_summary(recommendation))


# In[176]:


ifunc.get_summary(recommendation)


# In[ ]:




