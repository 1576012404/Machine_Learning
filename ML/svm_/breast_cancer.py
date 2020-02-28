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

plt.scatter(x[:,0],x[:,1])
plt.show()

from sklearn.preprocessing import StandardScaler
x=StandardScaler().fit_transform(x)

xtrain,xtest,ytrain,ytest=train_test_split(x,y,test_size=0.3,random_state=420)

Kernel=["linear","poly","rbf","sigmoid"]
for kernel in Kernel:
    time0=time()
    clf=SVC(kernel=kernel,gamma="auto",
            degree=1,
            cache_size=10000).fit(xtrain,ytrain)

    print("acc",kernel,clf.score(xtest,ytest))
    print("time cost:",time()-time0)










