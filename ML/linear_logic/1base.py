from sklearn.linear_model import LogisticRegression as LR
from sklearn.datasets import load_breast_cancer
import numpy as np
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score

data=load_breast_cancer()
x=data.data
y=data.target
lr1=LR(penalty="l1",solver="liblinear",C=0.5,max_iter=1000)
lr2=LR(penalty="l2",solver="liblinear",C=0.5,max_iter=1000)

lr1=lr1.fit(x,y)
print("coef",lr1.coef_)

lr2=lr2.fit(x,y)
print("coef2",lr2.coef_)

l1=[]
l2=[]
l1test=[]
l2test=[]

xtrain,xtest,ytrain,ytest=train_test_split(x,y,test_size=0.3,random_state=420)
for i in np.linspace(0.05,1.5,19):
    lr1=LR(penalty="l1",solver="liblinear",C=i,max_iter=1000)
    lr2=LR(penalty="l2",solver="liblinear",C=i,max_iter=1000)

    lr1=lr1.fit(xtrain,ytrain)
    l1.append(accuracy_score(lr1.predict(xtrain),ytrain))
    l1test.append(accuracy_score(lr1.predict(xtest),ytest))

    lr2=lr2.fit(xtrain,ytrain)
    l2.append(accuracy_score(lr2.predict(xtrain,),ytrain))
    l2test.append(accuracy_score(lr2.predict(xtest),ytest))

graph=[l1,l2,l1test,l2test]
color=["green","black","lightgreen","gray"]
label=["l1","l2","l1test","l2test"]
plt.figure(figsize=(6,6))
for i in range(len(graph)):
    plt.plot(np.linspace(0.05,1.5,19),graph[i],color[i],label=label[i])
plt.legend(loc=4)
plt.show()
















