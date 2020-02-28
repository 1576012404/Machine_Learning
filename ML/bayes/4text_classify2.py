from sklearn.datasets import fetch_20newsgroups
from sklearn.feature_extraction.text import TfidfVectorizer as TFIDF
import numpy as np

data=fetch_20newsgroups()


names=data.target_names
print("name",names)

categories = ["sci.space" #科学技术 - 太空
              # ,"rec.sport.hockey" #运动 - 曲棍球
              # ,"talk.politics.guns" #政治 - 枪支问题
              ,"talk.politics.mideast"] #政治 - 中东问题


train=fetch_20newsgroups(subset="train",categories=categories)
test=fetch_20newsgroups(subset="test",categories=categories)


xtrain=train.data
xtest=test.data
print("train",len(data))
# print("text",data[0])
ytrain=train.target
ytest=test.target

tfidf=TFIDF().fit(xtrain)
xtrain_=tfidf.transform(xtrain)
xtest_=tfidf.transform(xtest)
print("xtrain_",xtrain_.shape)


from sklearn.naive_bayes import MultinomialNB,ComplementNB,BernoulliNB
from sklearn.metrics import brier_score_loss as BS

name=["Multinomial","Complement","Bournulli"]
models=[MultinomialNB(),ComplementNB(),BernoulliNB()]


# for name,clf in zip(name,models):
#     clf.fit(xtrain_,ytrain)
#     y_pred=clf.predict(xtest_)
#     proba=clf.predict_proba(xtest_)
#     score=clf.score(xtest_,ytest)
#     print("<<<<<<<<name:",name,proba.shape)
#     bscore=[]
#     for i in range(len(np.unique(ytrain))):
#         print("i",i)
#         bs=BS(ytest,proba[:,i],pos_label=i)
#         bscore.append(bs)
#         print("label:",train.target_names[i],bs)
#     print("mean_bscore",np.mean(bscore))


for name, clf in zip(name, models):
    clf.fit(xtrain_, ytrain)
    y_pred = clf.predict(xtest_)
    proba = clf.predict_proba(xtest_)
    score = clf.score(xtest_, ytest)
    print(name)

    # 4个不同的标签取值下的布里尔分数
    Bscore = []
    for i in range(len(np.unique(ytrain))):
        bs = BS(ytest, proba[:, i], pos_label=i)
        Bscore.append(bs)
        print("\tBrier under {}:{:.3f}".format(train.target_names[i], bs))

    print("\tAverage Brier:{:.3f}".format(np.mean(Bscore)))
    print("\tAccuracy:{:.3f}".format(score))
    print("\n")





