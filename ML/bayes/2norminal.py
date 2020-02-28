from sklearn.preprocessing import MinMaxScaler
from sklearn.naive_bayes import MultinomialNB

from sklearn.model_selection import train_test_split
from sklearn.datasets import make_blobs
from sklearn.metrics import brier_score_loss

class_1=500
class_2=500
centers=[[0.0,0.0],[2.0,2.0]]
cluster_std=[0.5,0.5]
x,y=make_blobs(n_samples=[class_1,class_2],centers=centers,cluster_std=cluster_std,
               random_state=0,shuffle=False)

xtrain,xtest,ytrain,ytest=train_test_split(x,y,test_size=0.3,random_state=420)
mms=MinMaxScaler().fit(xtrain)
xtrain=mms.transform(xtrain)
xtest=mms.transform(xtest)


mnb=MultinomialNB().fit(xtrain,ytrain)

x=mnb.feature_log_prob_

print("x",x.shape,x)

score=mnb.score(xtest,ytest)
print("score",score)

from sklearn.preprocessing import KBinsDiscretizer
kbs=KBinsDiscretizer(n_bins=10,encode="onehot").fit(xtrain)
xtrain_=kbs.transform(xtrain)
xtest_=kbs.transform(xtest)
mnb=MultinomialNB().fit(xtrain_,ytrain)
score=mnb.score(xtest_,ytest)
print("score2",score)













