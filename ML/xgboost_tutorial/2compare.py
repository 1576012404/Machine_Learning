from xgboost import XGBRegressor as XGBR
import matplotlib.pyplot as plt
import numpy as np
from sklearn.datasets import load_boston
from sklearn.model_selection import KFold,cross_val_score as CVS,train_test_split as TTS
from sklearn.metrics import mean_squared_error as MSE
from sklearn.metrics import r2_score
data=load_boston()
x=data.data
y=data.target
xtrain,xtest,ytrain,ytest=TTS(x,y,test_size=0.3,random_state=420)

#
# reg=XGBR(n_estimators=180,random_state=420).fit(xtrain,ytrain)
# r21=reg.score(xtest,ytest)
# mes1=MSE(ytest,reg.predict(xtest))
#
# r22=r2_score(ytest,reg.predict(xtest))
# mse2=MSE(ytest,reg.predict(xtest))
#
# print("score",r21,mes1,r22,mse2)
#
# import xgboost as xgb
# dtrain=xgb.DMatrix(xtrain,ytrain)
# dtest=xgb.DMatrix(xtest,ytest)
# param={"silent":False,"objective":"reg:linear","eta":0.1}
# num_round=180
# bst=xgb.train(param,dtrain,num_round)
#
#
# r2=r2_score(ytest,bst.predict(dtest))
# mse=MSE(ytest,bst.predict(dtest))
# print("w2",r2,mse)





































