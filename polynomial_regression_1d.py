#!/usr/bin/env python

import assignment1 as a1
import numpy as np
import matplotlib.pyplot as plt

(countries, features, values) = a1.load_unicef_data()

targets = values[:,1]
x1 = values[:,7:15]
#x = a1.normalize_data(x)

N_TRAIN = 100 
N_TEST = 95
RMS_list_train = list()
RMS_list_test = list()
W_list = list()
x1[:2,3]
for j in range(1, 9):
    x = x1[:,j-1:j]
    x_train = x[0:N_TRAIN,:]
    x_test = x[N_TRAIN:,:]
    t_train = targets[0:N_TRAIN]
    t_test = targets[N_TRAIN:]
    Desmat_train = np.ones((N_TRAIN,1),dtype=np.int32)
    Desmat_test = np.ones((N_TEST,1),dtype=np.int32)
    
    
    for i in range(1,4):
        Desmat_train = np.concatenate((Desmat_train,np.power(x_train,i)),axis=1)
        Desmat_test = np.concatenate((Desmat_test,np.power(x_test,i)),axis=1)
    W = np.linalg.pinv(Desmat_train)*t_train
    W_list.append(W)
    Y_train = Desmat_train*W
    E_train = np.square(Y_train - t_train)
    RMS_train = np.sqrt(np.sum(E_train)/N_TRAIN)
    RMS_list_train.append(RMS_train)
    Y_test = Desmat_test*W
    E_test = np.square(Y_test - t_test)
    RMS_test = np.sqrt(np.sum(E_test)/N_TEST)
    RMS_list_test.append(RMS_test)


#Produce a plot of results.
plt.bar(np.arange(8,16), RMS_list_train,0.25, color='red')
plt.bar(np.arange(8,16)+0.25,RMS_list_test,0.25)
#plt.plot(train_err.keys(), train_err.values())
#plt.plot(test_err.keys(), test_err.values())
plt.ylabel('RMS')
plt.legend(['Training error','Test error'])
plt.title('Training and Test for features 8 to 15')
plt.xlabel('Features')
plt.show()
#

x_ev_list = list()
y_ev_list = list()

for i in range(3,6):
    x_ev = np.matrix(np.linspace(np.asscalar(min(x1[:100,i])), np.asscalar(max(x1[:100,i])), num=500)).transpose()
    x_ev_list.append(x_ev)
    Mat_one = np.ones((500,1),dtype=np.int32)
    y_ev = np.concatenate((Mat_one,x_ev),axis=1)
    y_ev = np.concatenate((y_ev,np.power(x_ev,2)),axis=1)
    y_ev = np.concatenate((y_ev,np.power(x_ev,3)),axis=1)
    y_ev = y_ev * W_list[i]
    y_ev_list.append(y_ev)
    plt.plot(x1[:100,i],'go')   
    plt.plot(x1[100:,i],'bo')
    plt.plot(y_ev_list[i-3],x_ev_list[i-3],'r-')
    plt.legend(['Training data points','Test data points', 'Learned polynomial'])
    plt.xlabel('Feature Value')   
    plt.ylabel('Target Value')
    plt.show()