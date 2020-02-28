from sklearn.linear_model import LinearRegression as LR
from sklearn.model_selection import train_test_split
from sklearn.model_selection import cross_val_score
import numpy as np

xtrain=np.arange(6).reshape(3,2)
ytrain=np.array([1,0,1])
xtest=np.arange(100,106).reshape(3,2)
ytest=np.array([0,0,1])

print("xtrain",xtrain)


reg=LR().fit(xtrain,ytrain)
yhat=reg.predict(xtest)



print("w",xtest.shape)
print("coef",reg.coef_)

from sklearn.metrics import mean_squared_error as MSE
mse=MSE(yhat,ytest)
print("mes",mse)
print("mean",ytest.mean())

# s=cross_val_score(reg,X,y,cv=10,scoring="neg_mean_squared_error")
# print("s",s)
# print("s.mean",s.mean())