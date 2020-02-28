from sklearn.linear_model import LinearRegression as LR
from sklearn.model_selection import train_test_split
from sklearn.model_selection import cross_val_score
# from sklearn.datasets import fetch_california_housing as fch
import pandas as pd
from sklearn.datasets import load_boston
from sklearn.datasets import load_boston
from sklearn.model_selection import KFold,cross_val_score as CVS,train_test_split as TTS
data=load_boston()
x=data.data
y=data.target
print("x",x.shape)
print("y",y.shape)

X=pd.DataFrame(x)
xtrain,xtest,ytrain,ytest=TTS(X,y,test_size=0.3,random_state=420)
for i in [xtrain,xtest]:
    i.index=range(i.shape[0])

reg=LR().fit(xtrain,ytrain)
yhat=reg.predict(xtest)
print("w",xtest.shape)
print("coef",reg.coef_)

from sklearn.metrics import mean_squared_error as MSE
mse=MSE(yhat,ytest)
print("mes",mse)
print("mean",ytest.mean())

s=cross_val_score(reg,X,y,cv=10,scoring="neg_mean_squared_error")
print("s",s)
print("s.mean",s.mean())