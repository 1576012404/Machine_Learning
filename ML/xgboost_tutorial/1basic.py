from xgboost import XGBRegressor as XGBR
from sklearn.ensemble import RandomForestRegressor as RFR
from sklearn.linear_model import LinearRegression

from sklearn.datasets import load_boston
from sklearn.model_selection import KFold,cross_val_score as CVS,train_test_split as TTS
from sklearn.metrics import mean_squared_error as MSE
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from time import time
import datetime

data=load_boston()

x=data.data
y=data.target

xtrain,xtest,ytrain,ytest=TTS(x,y,test_size=0.3,random_state=420)

reg=XGBR(n_estimators=100).fit(xtrain,ytrain)

reg.predict(xtest)
reg.score(xtest,ytest)

err=MSE(ytest,reg.predict(xtest))
ipt=reg.feature_importances_

# print("err",err)
# print("ipt",ipt)



reg=XGBR(n_estimators=100)
an=CVS(reg,xtrain,ytrain,cv=5).mean()
print("an",an)

an2=CVS(reg,xtrain,ytrain,cv=5,scoring="neg_mean_squared_error").mean()
print("an2",an2)


rfr=RFR(n_estimators=100)
a=CVS(rfr,xtrain,ytrain,cv=5).mean()
neg_mean_square=CVS(rfr,xtrain,ytrain,scoring="neg_mean_squared_error").mean()
print("a,",a,neg_mean_square)


lr=LinearRegression()
b=CVS(lr,xtrain,ytrain,cv=5).mean()
bb=CVS(lr,xtrain,ytrain,cv=5,scoring="neg_mean_squared_error").mean()
print("b",b,bb)

reg=XGBR(n_estimators=100,silent=False)
c=CVS(reg,xtrain,ytrain,cv=5,scoring="neg_mean_squared_error").mean()
print("c",c)





































