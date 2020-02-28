from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.datasets import load_wine
from sklearn.model_selection  import train_test_split


wine=load_wine()
xtrain,xtest,ytrain,ytest=train_test_split(wine.data,wine.target,
                                           test_size=0.3)

# clf=DecisionTreeClassifier(random_state=0)
# rfc=RandomForestClassifier(random_state=0)
# clf=clf.fit(xtrain,ytrain)
# rfc=rfc.fit(xtest,ytest)
# score_c=clf.score(xtest,ytest)
# score_r=rfc.score(xtest,ytest)
# print("tree",score_c,"forest",score_r
#       )



from sklearn.model_selection import cross_val_score
import matplotlib.pyplot as plt

rfc=RandomForestClassifier(n_estimators=25)
rfc_s=cross_val_score(rfc,wine.data,wine.target,cv=10)
print("rfc_s",rfc_s)

clf=DecisionTreeClassifier()

clf_s=cross_val_score(clf,wine.data,wine.target,cv=10)

plt.plot(range(1,11),rfc_s,label="randomforest")
plt.plot(range(1,11),clf_s,label="decision tree")
plt.legend()
plt.show()



