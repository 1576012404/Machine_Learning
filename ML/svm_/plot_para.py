from sklearn.datasets import load_breast_cancer
from sklearn.svm import SVC
from sklearn.model_selection import train_test_split
import matplotlib.pyplot as plt
import numpy as np
from time import time
import datetime

data=load_breast_cancer()
x=data.data
y=data.target
print("x",x.shape,y.shape)
class_=np.unique(y)
print("class",class_)

# plt.scatter(x[:,0],x[:,1])
# plt.show()

from sklearn.preprocessing import StandardScaler
x=StandardScaler().fit_transform(x)

xtrain,xtest,ytrain,ytest=train_test_split(x,y,test_size=0.3,random_state=420)
score=[]
gamma_range=np.logspace(-10,1,50)
for i in gamma_range:
    clf=SVC(kernel="rbf",gamma=i,cache_size=5000).fit(xtrain,ytrain)
    score.append(clf.score(xtest,ytest))

print(max(score),gamma_range[score.index(max(score))])
plt.plot(gamma_range,score)
plt.show()