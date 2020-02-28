from sklearn.preprocessing import MinMaxScaler
from sklearn.naive_bayes import MultinomialNB,BernoulliNB

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
xtrain_=mms.transform(xtrain)
xtest_=mms.transform(xtest)
bnl=BernoulliNB().fit(xtrain_,ytrain)
score=bnl.score(xtest_,ytest)
print("score",score)

bnl=BernoulliNB(binarize=0.5).fit(xtrain_,ytrain)
s=bnl.score(xtest_,ytest)
print("s",s)

