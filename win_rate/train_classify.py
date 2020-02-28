from sklearn.linear_model import LogisticRegression as LR
from sklearn.preprocessing import StandardScaler
import numpy as np
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score
import pandas as pd

from sklearn.preprocessing import StandardScaler


def Train():
    filename="train_classify.csv"
    data=pd.read_csv(filename,index_col=0,nrows=10)
    float_cols = [c for c in data if data[c].dtype == "float64"]
    float32_cols = {c: np.float32 for c in float_cols}
    data = pd.read_csv(filename, index_col=0, dtype=float32_cols)
    x = data.values[:,:-1]
    y = data.label

    scaler=StandardScaler()
    scaler.fit(x)
    x = scaler.transform(x)



    print("x",x.shape)
    xtrain, xtest, ytrain, ytest = train_test_split(x, y, test_size=0.3, random_state=420)


    l1 = []
    l2 = []
    l1test = []
    l2test = []
    i_list=np.linspace(0.05, 1.5, 10)
    for i in i_list:
        lr1 = LR(penalty="l1", solver="liblinear", C=i, max_iter=1000)
        lr2 = LR(penalty="l2", solver="liblinear", C=i, max_iter=1000)
        lr1 = lr1.fit(xtrain, ytrain)
        l1.append(accuracy_score(lr1.predict(xtrain), ytrain))
        l1_score=accuracy_score(lr1.predict(xtest), ytest)
        print("l1_score",l1_score)
        l1test.append(l1_score)
        lr2 = lr2.fit(xtrain, ytrain)
        l2.append(accuracy_score(lr2.predict(xtrain, ), ytrain))
        l2_score=accuracy_score(lr2.predict(xtest), ytest)
        print("l2_score", l2_score)
        l2test.append(l2_score)
    l1_max=max(l1test)
    l1_index=l1test.index(l1_max)
    print("l1",l1_max,l1_index)
    l2_max = max(l2test)
    l2_index = l2test.index(l2_max)
    print("l2", l2_max, l2_index)





    # i = 0.5
    # lr2 = LR(penalty="l2", solver="liblinear", C=i, max_iter=1000)
    # lr2 = lr2.fit(xtrain, ytrain)
    # acc1 = accuracy_score(lr2.predict(xtrain ), ytrain)
    # acc2 = accuracy_score(lr2.predict(xtest), ytest)
    # print("acc", acc1, acc2)



if __name__=="__main__":
    Train()
