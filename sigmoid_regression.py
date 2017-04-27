#!/usr/bin/env python

import assignment1 as a1
import numpy as np
import matplotlib.pyplot as plt

def sigmoid_f(u,x,s):
    return 1.0/(1.0+np.exp((u-x)/s))

(countries, features, values) = a1.load_unicef_data()

targets = values[:,1]
x = values[:,10:11]
#x = a1.normalize_data(x)

N_TRAIN = 100
N_TEST = 95
S = 2000.0

x_train = x[0:N_TRAIN,:]
x_test = x[N_TRAIN:,:]
t_train = targets[0:N_TRAIN]
t_test = targets[N_TRAIN:]


Desmat_train = np.ones((N_TRAIN,1),dtype=np.int32)
Desmat_test = np.ones((N_TEST,1),dtype=np.int32)
RMS_list_train = {}
RMS_list_test = {}


Desmat_train = np.concatenate((Desmat_train,sigmoid_f(100,x_train,S)),axis=1)
Desmat_train = np.concatenate((Desmat_train,sigmoid_f(10000,x_train,S)),axis=1)

W = np.linalg.pinv(Desmat_train)*t_train
Y_train = Desmat_train*W
E_train = np.square(Y_train - t_train)

RMS_train = np.sqrt(np.sum(E_train)/N_TRAIN)


Desmat_test = np.concatenate((Desmat_test,sigmoid_f(100,x_test,S)),axis=1)
Desmat_test = np.concatenate((Desmat_test,sigmoid_f(10000,x_test,S)),axis=1)

Y_test = Desmat_test*W
E_test = np.square(Y_test - t_test)

RMS_test = np.sqrt(np.sum(E_test)/N_TEST)


W = np.squeeze(np.asarray(W))

x_ev = np.linspace(np.asscalar(min(x_train)), np.asscalar(max(x_train)), num=500)

# TO DO:: Put your regression estimate here in place of x_ev.
# Evaluate regression on the linspace samples.
y_ev = (np.ones(x_ev.shape)*W[0]+sigmoid_f(100,x_ev,S)*W[1]+sigmoid_f(10000,x_ev,S)*W[2])

plt.plot(x_ev,y_ev,'r.-')
plt.plot(x_train,t_train,'bo')
plt.title('Sigmoidal Basis Function')
plt.show()
