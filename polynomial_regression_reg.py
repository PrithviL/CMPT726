#!/usr/bin/env python

import assignment1 as a1
import numpy as np
import matplotlib.pyplot as plt
#import matplotlib.pyplot.semilogx 

(countries, features, values) = a1.load_unicef_data()

targets = values[:,1]
x = values[:,7:]
#x = a1.normalize_data(x)

N_TRAIN = 100;
x_train = x[0:N_TRAIN,:]
x_test = x[N_TRAIN:,:]
t_train = targets[0:N_TRAIN]
t_test = targets[N_TRAIN:]

e_list_train=list()
e_list_val=list()
#e_list_test=list()

#lambda_values=list()
lambda_values=[0,0.01,0.1,1,10,10**2,10**3,10**4]

#for i in range(1,3):
#    desm=np.concatenate((desm,np.power(x_train,i)),axis=1)
#    desm_test=np.concatenate((desm_test,np.power(x_test,i)),axis=1)
#w=np.linalg.pinv(dm)*t_train
#Y_train=dm*w
e_avg_rms_train = []
e_avg_rms_val = []
#.Y_test=desm_test*w
for j in lambda_values:
    
    e_list_train = []    
    e_list_val = []
    for i in range(1,11):
        
        t_matrix=np.concatenate((t_train[:(i-1)*10],t_train[i*10:]),axis=0)

        f_matrix_train=np.concatenate((x_train[:(i-1)*10],x_train[i*10:]),axis=0) 
        f_matrix_val = x_train[(i-1)*10:i*10]

        v_matrix=t_train[(i-1)*10:i*10]

        desm = np.ones((N_TRAIN-10,1),np.int)
        desm_val=np.ones((N_TRAIN-90,1),np.int)
        #sprint v_matrix
        for r in range(1,3):            
            desm=np.concatenate((desm,np.power(f_matrix_train,r)),axis=1)            
            desm_val=np.concatenate((desm_val,np.power(f_matrix_val,r)),axis=1)
        
        w = np.linalg.inv(j*np.identity(67) + desm.transpose()*desm) * desm.transpose()*t_matrix
        
        Y_train=desm*w
        E_train=np.square(Y_train-t_matrix)

        Erms_train= np.sqrt(np.sum(E_train)/90)
        Y_val=desm_val*w        
        E_val=np.square(Y_val-v_matrix)
        Erms_val=np.sqrt(np.sum(E_val)/10)

        e_list_train.append(Erms_train)
        e_list_val.append(Erms_val)
        
    e_avg_rms_train.append(np.mean(e_list_train))
    e_avg_rms_val.append(np.mean(e_list_val))
    
   
#    E_test=(np.square(Y_train-t_test)+((lambda_values[j]/2)*np.square(w)))
#    Erms_test= np.sqrt(np.sum(E_test)/N_test)
#    e_list_test.append(Erms_test)
#print e_list_test
# Produce a plot of results.
#plt.plot([1,2,3,4,5,6], e_list_train)
#plt.plot([1,2,3,4,5,6], e_list_test)
#plt.ylabel('RMS')
#plt.legend(['Training error','Validation error'])
#plt.title('Regularized Polynomial Regression')
#plt.xlabel('Polynomial degree')
#plt.show()

#plt.semilogx(lambda_values, e_avg_rms_train,color='Blue')
plt.semilogx(lambda_values,e_avg_rms_val,color='green')
plt.ylabel('RMS')
plt.legend(['Training error','Validation error'])
plt.title('Regularized Polynomial Regression')
plt.xlabel('Features')
plt.show()