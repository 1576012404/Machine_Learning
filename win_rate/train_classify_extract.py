from sklearn.linear_model import LogisticRegression as LR
from sklearn.preprocessing import StandardScaler
import numpy as np
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score
import pandas as pd

from sklearn.preprocessing import StandardScaler
from sklearn.feature_selection import VarianceThreshold

def Train():
    filename="train_classify.csv"
    data=pd.read_csv(filename,index_col=0,nrows=10)
    float_cols = [c for c in data if data[c].dtype == "float64"]
    float32_cols = {c: np.float32 for c in float_cols}
    data = pd.read_csv(filename, index_col=0, dtype=float32_cols)
    x = data.values[:,:-1]
    y = data.label
    print("pre",x.shape)
    scaler = VarianceThreshold()
    scaler.fit(x)
    x = scaler.transform(x)
    scaler=StandardScaler()
    scaler.fit(x)
    x = scaler.transform(x)
    i=0.5
    print("x",x.shape)
    xtrain, xtest, ytrain, ytest = train_test_split(x, y, test_size=0.3, random_state=420)
    lr1 = LR(penalty="l1", solver="liblinear", C=i, max_iter=1000)
    lr2 = LR(penalty="l2", solver="liblinear", C=i, max_iter=1000)
    lr1 = lr1.fit(xtrain, ytrain)
    l1_score = accuracy_score(lr1.predict(xtest), ytest)
    print("l1_score", l1_score)
    lr2 = lr2.fit(xtrain, ytrain)
    l2_score = accuracy_score(lr2.predict(xtest), ytest)
    print("l2_score", l2_score)





    # i = 0.5
    # lr2 = LR(penalty="l2", solver="liblinear", C=i, max_iter=1000)
    # lr2 = lr2.fit(xtrain, ytrain)
    # acc1 = accuracy_score(lr2.predict(xtrain ), ytrain)
    # acc2 = accuracy_score(lr2.predict(xtest), ytest)
    # print("acc", acc1, acc2)



if __name__=="__main__":
    Train()
