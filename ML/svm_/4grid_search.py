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

from sklearn.model_selection import StratifiedShuffleSplit
from sklearn.model_selection import GridSearchCV

time0=time()
gamma_range=np.logspace(-10,1,20)
coef0_range=np.linspace(0,5,10)

param_grid=dict(gamma=gamma_range,coef0=coef0_range)
cv=StratifiedShuffleSplit(n_splits=5,test_size=0.3,random_state=420)
grid=GridSearchCV(SVC(kernel="poly",degree=1,cache_size=5000),param_grid=param_grid,cv=cv)
grid.fit(x,y)

print("best_para",grid.best_params_)
print("best_score",grid.best_score_)











