import numpy as np
import matplotlib.pyplot as plt
from sklearn.naive_bayes import GaussianNB
from sklearn.datasets import load_digits
from sklearn.model_selection import train_test_split

digits=load_digits()
x,y=digits.data,digits.target
xtrain,xtest,ytrain,ytest=train_test_split(x,y,test_size=0.3,random_state=420)
print("xtrain",xtrain.shape)

gnb=GaussianNB().fit(xtrain,ytrain)
acc_score=gnb.score(xtest,ytest)
print("acc_score",acc_score)
prob=gnb.predict_proba(xtest)
print("prob",xtest.shape,ytest.shape,prob.shape)

y_pred=gnb.predict(xtest)
from sklearn.metrics import confusion_matrix as CM
cm=CM(ytest,y_pred)
print("cm",cm)



