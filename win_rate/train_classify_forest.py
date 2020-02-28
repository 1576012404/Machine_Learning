from sklearn.linear_model import LogisticRegression as LR
from sklearn.preprocessing import StandardScaler
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier
import numpy as np
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score
import pandas as pd

from sklearn.preprocessing import StandardScaler
from sklearn.feature_selection import VarianceThreshold
from time import time
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

    xtrain, xtest, ytrain, ytest = train_test_split(x, y, test_size=0.3, random_state=420)

    rfc = RandomForestClassifier(n_estimators=100)
    rfc.fit(xtrain,ytrain)
    score = accuracy_score(rfc.predict(xtest), ytest)
    print("score",score)








if __name__=="__main__":
    t1=time()
    Train()
    print("cost",time()-t1)
