#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sat Nov 17 14:52:56 2018

@author: mridhoula
"""

import pickle
import gzip

filename = 'mnist.pkl.gz'
f = gzip.open(filename, 'rb')
training_data, validation_data, test_data = pickle.load(f, encoding='latin1')

x_train=training_data[0]
y_train=training_data[1]

validation_data=validation_data[0]
target_data1=validation_data[1]

x_test=test_data[0]
y_test=test_data[1]

f.close()


from PIL import Image
import os
import numpy as np

USPSMat  = []
USPSTar  = []
curPath  = 'USPSdata/Numerals'
savedImg = []

for j in range(0,10):
    curFolderPath = curPath + '/' + str(j)
    imgs =  os.listdir(curFolderPath)
    for img in imgs:
        curImg = curFolderPath + '/' + img
        if curImg[-3:] == 'png':
            img = Image.open(curImg,'r')
            img = img.resize((28, 28))
            savedImg = img
            imgdata = (255-np.array(img.getdata()))/255
            USPSMat.append(imgdata)
            USPSTar.append(j)
        
        
        
        
        
        
###Neural network for MNIST Dataset      
import keras
from keras.layers import Dense
from keras.models import Sequential
num_classes=10
image_vector_size=28*28
USPSMat_1=np.asarray(USPSMat)
USPSMat_1 = USPSMat_1.reshape(USPSMat_1.shape[0], image_vector_size)
USPSTar_1=np.asarray(USPSTar)
USPSTar_1 = keras.utils.to_categorical(USPSTar_1, num_classes)
x_train = x_train.reshape(x_train.shape[0], image_vector_size)
x_test = x_test.reshape(x_test.shape[0], image_vector_size)
y_train = keras.utils.to_categorical(y_train, num_classes)
y_test = keras.utils.to_categorical(y_test, num_classes)
image_size = 784
model = Sequential()
model.add(Dense(units=32, activation='sigmoid', input_shape=(image_size,)))
model.add(Dense(units=num_classes, activation='softmax'))
model.compile(optimizer='adam', loss='categorical_crossentropy',metrics=['accuracy'])
history = model.fit(x_train, y_train, batch_size=128, epochs=10,
verbose=False,validation_split=.1)
loss,accuracy = model.evaluate(x_test, y_test, verbose=False)
print("Neural network for MNIST Dataset")
print(accuracy)
loss1,accuracy1 = model.evaluate(USPSMat_1,USPSTar_1, verbose=False)
print("Neural network for USPS Dataset") 
print(accuracy1)

from sklearn.metrics import confusion_matrix
pred=model.predict(x_test)
pred=np.argmax(pred,axis=1)
y_test=np.argmax(y_test,axis=1)
g=confusion_matrix(y_test,pred)
print("Confusion matrix for  Neural Network: MNIST")
print(g)
pred1=model.predict(USPSMat_1)
pred1=np.argmax(pred1,axis=1)
USPSTar_1=np.argmax(USPSTar_1,axis=1)
h=confusion_matrix(USPSTar_1,pred1)
print("Confusion matrix for Neural Network:USPS")
print(h)


import numpy as np
from sklearn.svm import SVC
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score
x_train=training_data[0]
y_train=training_data[1]
x_test=test_data[0]
y_test=test_data[1]

print("SVMfor MNIST Dataset")
classifier1 = SVC(kernel='linear', gamma=1, C=2);
classifier1.fit(x_train, y_train )
y_pred_svm=classifier1.predict(x_test)
acc_svm=accuracy_score(y_test, y_pred_svm)
print(acc_svm)


print("Random Forest Classifier for MNIST Dataset")
classifier2 = RandomForestClassifier(n_estimators=10);
classifier2.fit(x_train, y_train)
y_pred_rf=classifier2.predict(x_test)
acc_rf=accuracy_score(y_test,y_pred_rf)
print(acc_rf)


#Softmax Regression for MNIST dataset
import numpy as np
import matplotlib.pyplot as plt
import scipy.sparse


def getLoss(w,x,y,lam):
    m = x.shape[0]
    y_mat = encoding(y) 
    scores = np.dot(x,w)
    prob = softmax(scores) 
    loss = (-1 / m) * np.sum(y_mat * np.log(prob)) + (lam/2)*np.sum(w*w) 
    grad = (-1 / m) * np.dot(x.T,(y_mat - prob)) + lam*w 
    return loss,grad

def encoding(Y):
    m = Y.shape[0]
    #Y = Y[:,0]
    h = scipy.sparse.csr_matrix((np.ones(m), (Y, np.array(range(m)))))
    h = np.array(h.todense()).T
    return h

def softmax(z):
    z -= np.max(z)
    sm = (np.exp(z).T / np.sum(np.exp(z),axis=1)).T
    return sm

def prediction(X):
    probs = softmax(np.dot(X,w))
    preds = np.argmax(probs,axis=1)
    return probs,preds

w = np.zeros([x_train.shape[1],len(np.unique(y_train))])
lam = 0.01
iterations = 1000
learningRate = 0.01
losses = []
for i in range(0,iterations):
    loss,grad = getLoss(w,x_train,y_train,lam)
    losses.append(loss)
    w = w - (learningRate * grad)


plt.plot(losses)

def getAccuracy(x,y):
    prob,prede = prediction(x)
    accuracy = sum(prede == y)/(float(len(y)))
    return accuracy
print("Softmax Regression for MNIST dataset")
print ('Training Accuracy: ', getAccuracy(x_train,y_train))
print ('Test Accuracy: ', getAccuracy(x_test,y_test))





USPSMat_1=np.asarray(USPSMat)
USPSTar_1=np.asarray(USPSTar)
def getLoss(w,x,y,lam):
    m = x.shape[0] #First we get the number of training examples
    y_mat = encoding(y) #Next we convert the integer class coding into a one-hot representation
    scores = np.dot(x,w) #Then we compute raw class scores given our input and current weights
    prob = softmax(scores) #Next we perform a softmax on these scores to get their probabilities
    loss = (-1 / m) * np.sum(y_mat * np.log(prob)) + (lam/2)*np.sum(w*w) #We then find the loss of the probabilities
    grad = (-1 / m) * np.dot(x.T,(y_mat - prob)) + lam*w #And compute the gradient for that loss
    return loss,grad

def encoding(Y):
    m = Y.shape[0]
    #Y = Y[:,0]
    h = scipy.sparse.csr_matrix((np.ones(m), (Y, np.array(range(m)))))
    h = np.array(h.todense()).T
    return h

def softmax(z):
    z -= np.max(z)
    sm = (np.exp(z).T / np.sum(np.exp(z),axis=1)).T
    return sm

def prediction(X):
    probs = softmax(np.dot(X,w))
    preds = np.argmax(probs,axis=1)
    return probs,preds

w = np.zeros([x_train.shape[1],len(np.unique(y_train))])
lam = 0.01
iterations = 1000
learningRate = 0.01
losses = []
for i in range(0,iterations):
    loss,grad = getLoss(w,x_train,y_train,lam)
    losses.append(loss)
    w = w - (learningRate * grad)


plt.plot(losses)

def getAccuracy(x,y):
    prob,prede1 = prediction(x)
    accuracy = sum(prede1 == y)/(float(len(y)))
    return accuracy
print("Softmax Regression for USPS dataset")
print ('Test Accuracy: ', getAccuracy(USPSMat_1,USPSTar_1))


print("SVM Classifier for USPS: dataset")
USPSMat_svm=np.asarray(USPSMat)
USPSTar_svm=np.asarray(USPSTar)
classifier1 = SVC(kernel='linear', gamma=1 , C=2);
classifier1.fit(x_train,y_train )
y_pred_svm_usps=classifier1.predict(USPSMat_svm)
acc_svm=accuracy_score(USPSTar_svm, y_pred_svm_usps)
print(acc_svm)


print("Random Forest Classifier for USPS :Dataset")
USPSMat_rf=np.asarray(USPSMat)
USPS_Tar_rf=np.asarray(USPSTar)
classifier2 = RandomForestClassifier(n_estimators=10);
classifier2.fit(x_train,y_train)
y_pred_rf_usps=classifier2.predict(USPSMat_rf)
acc_rf=accuracy_score(USPS_Tar_rf,y_pred_rf_usps)
print(acc_rf)


from sklearn.metrics import confusion_matrix
a=confusion_matrix(y_test,y_pred_svm)
print("Confusion matrix for SVM: MNIST")
print(a)
b=confusion_matrix(USPSTar_svm, y_pred_svm_usps)
print("Confusion matrix for SVM:  USPS")
print(b)


from sklearn.metrics import confusion_matrix
c=confusion_matrix(y_test,y_pred_rf)
print("Confusion matrix for  Random Forest: MNIST")
print(c)
d=confusion_matrix(USPS_Tar_rf,y_pred_rf_usps)
print("Confusion matrix for Random forest:USPS")
print(d)


from sklearn.metrics import confusion_matrix
probs,prede=prediction(x_test)
e=confusion_matrix(y_test,prede)
print("Confusion matrix for Logistic regression: MNIST")
print(e)
probs1,prede1=prediction(USPSMat_1)
f=confusion_matrix(USPSTar_1,prede1)
print("Confusion matrix for Logistic Regression:  USPS")
print(f)

from  sklearn.ensemble import VotingClassifier

from sklearn.linear_model import LogisticRegression
x_train=training_data[0]
y_train=training_data[1]
x_test=test_data[0]
y_test=test_data[1]
rf=RandomForestClassifier()
lr=LogisticRegression()
svm=SVC(kernel='rbf',degree=2)
evc=VotingClassifier(estimators=[('lr',lr),('rf',rf),('svm',svm)],voting='hard')
evc.fit(x_train,y_train)
a=evc.score(x_test,y_test)
print("Combined model accuracy using majority voting")
print(a)


    


