#!/usr/bin/env python

import assignment1 as a1
import numpy as np
import matplotlib.pyplot as plt

(countries, features, values) = a1.load_unicef_data()

targets = values[:,1]
x = values[:,7:]
x = a1.normalize_data(x)

N_TRAIN = 100;
x_train = x[0:N_TRAIN,:]
x_test = x[N_TRAIN:,:]
t_train = targets[0:N_TRAIN]
t_test = targets[N_TRAIN:]
N_TEST=95;

Desmat_train=np.ones((N_TRAIN,1),dtype=np.int32)
Desmat_test=np.ones((N_TEST,1),dtype=np.int32)
RMS_list_train={}
RMS_list_test={}

#w is weight

for i in range(1,7):
    Desmat_train=np.concatenate((Desmat_train,np.power(x_train,i)),axis=1)
    W=np.linalg.pinv(Desmat_train)*t_train
    
    Y_train=Desmat_train*W
    E_train=np.square(Y_train - t_train)
    
    RMS_train=np.sqrt(np.sum(E_train)/N_TRAIN)
    RMS_list_train[i]=RMS_train
    
    Desmat_test=np.concatenate((Desmat_test,np.power(x_test,i)),axis=1)
    
    Y_test=Desmat_test * W
    E_test=np.square(Y_test - t_test)
    
    RMS_test=np.sqrt(np.sum(E_test)/N_TEST)
    RMS_list_test[i]=RMS_test

# Produce a plot of results.
plt.plot(RMS_list_train.keys(),RMS_list_train.values())
plt.plot(RMS_list_test.keys(),RMS_list_test.values())
plt.ylabel('RMS')
plt.legend(['Test error','Training error'])
plt.title('Fit with polynomials')
plt.xlabel('Polynomial degree')
plt.show()
