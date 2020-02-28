from sklearn.model_selection import learning_curve
from xgboost import XGBRegressor as XGBR
import matplotlib.pyplot as plt
import numpy as np
from sklearn.datasets import load_boston
from sklearn.model_selection import KFold,cross_val_score as CVS,train_test_split as TTS
# def plot_learning_curve(estimator,title,x,y,
#                         ax=None,
#                         ylim=None,
#                         cv=None,
#                         n_jobs=None):
#     train_sizes,train_scores,test_scores=learning_curve(
#         estimator,x,y,shuffle=True,cv=cv,n_jobs=n_jobs
#     )
#     if ax==None:
#         ax=plt.gca()
#     else:
#         ax=plt.figure()
#     ax.set_title(title)
#     if ylim is not None:
#         ax.set_ylim(*ylim)
#     ax.set_xlabel("training examples")
#     ax.set_ylabel("score")
#     ax.grid()
#     ax.plot(train_sizes,np.mean(train_scores,axis=1),"o-",color="r",label="training score")
#     ax.plot(train_sizes,np.mean(test_scores,axis=1),"o-",color="g",label="test score")
#     ax.legend(loc="best")
#     return ax
#
#
# data=load_boston()
#
# x=data.data
# y=data.target
#
# xtrain,xtest,ytrain,ytest=TTS(x,y,test_size=0.3,random_state=420)
# cv=KFold(n_splits=5,shuffle=True,random_state=42)
# plot_learning_curve(XGBR(n_estimators=100,random_state=420),"XGB",xtrain,ytrain,ax=None,cv=cv)
# plt.show()




data=load_boston()
x=data.data
y=data.target
xtrain,xtest,ytrain,ytest=TTS(x,y,test_size=0.3,random_state=420)




# rs=[]
# var=[]
# ge=[]
# axisx=range(100,300,10)
# for i in range(100,300,10):
#     reg=XGBR(n_estimators=i,random_state=420)
#     cvresult=CVS(reg,xtrain,ytrain,cv=5)
#     rs.append(cvresult.mean())
#     var.append(cvresult.var())
#     ge.append((1-cvresult.mean())**2+cvresult.var())
#
# print(axisx[rs.index(max(rs))],max(rs),var[rs.index(max(rs))])
# print(axisx[var.index(min(var))],rs[var.index(min(var))],min(var))
# print(axisx[ge.index(min(ge))],rs[ge.index(min(ge))],var[ge.index(min(ge))],min(ge))
#
#
# rs=np.array(rs)
# var=np.array(var)*0.01
# plt.figure(figsize=(20,5))
# plt.plot(axisx,rs,c="black",label="XGB")
# plt.plot(axisx,rs+var,c="red",linestyle="-.")
# plt.plot(axisx,rs-var,c="red",linestyle="-.")
# plt.legend()
# plt.show()




# axisx = range(10,1010,50)
# rs = []
# for i in axisx:
#    reg = XGBR(n_estimators=i,random_state=420)
#    rs.append(CVS(reg,xtrain,ytrain,cv=5).mean())
# print(axisx[rs.index(max(rs))],max(rs))
# plt.figure(figsize=(20,5))
# plt.plot(axisx,rs,c="red",label="XGB")
# plt.legend()
# plt.show()

def regassess(reg,xtrain,ytrain,cv,scoring=["r2"],show=True):
    score=[]
    for i in range(len(scoring)):
        s=CVS(reg,xtrain,ytrain,cv=5,scoring=scoring[i]).mean()
        if show:
            print("score",i,s)
        score.append(s)
    return score
from time import time
for i in [0,0.2,0.5,1]:
    reg=XGBR(n_estimators=180,random_state=420,learning_rate=i)
    print("learning_rate",i)
    t1=time()
    regassess(reg,xtrain,ytrain,4,scoring=["r2","neg_mean_squared_error"])
    # print("cost",time()-t1)





















