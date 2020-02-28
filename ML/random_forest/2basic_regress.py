from sklearn.datasets import load_breast_cancer
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import GridSearchCV,cross_val_score
import matplotlib.pyplot as plt
import pandas as pd
import numpy as np


data=load_breast_cancer()

#score_pre 0.9666925935528475
# rfc=RandomForestClassifier(n_estimators=100,random_state=90)
# score_pre=cross_val_score(rfc,data.data,data.target,cv=10).mean()
# print("score_pre",score_pre)




# scorel=[]
# for i in range(0,200,10):
#     rfc=RandomForestClassifier(n_estimators=i+1,
#                                n_jobs=-1,
#                                random_state=90
#
#                                )
#     score=cross_val_score(rfc,data.data,data.target,
#                           cv=10).mean()
#     scorel.append(score)

# print("max",max(scorel),)
# plt.figure(figsize=[20,5])
# plt.plot(range(1,201,10),scorel)
# plt.show()





param_grid={"max_features":np.arange(5,30,1)}
rfc=RandomForestClassifier(n_estimators=39,
                           random_state=90)
GS=GridSearchCV(rfc,param_grid,cv=2)
GS.fit(data.data,data.target)

print("Para",GS.best_params_)
print("score",GS.best_score_F)
















