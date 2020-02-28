from sklearn.linear_model import LogisticRegression as LR
from sklearn.preprocessing import StandardScaler
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.ensemble import RandomForestRegressor

from sklearn.metrics import mean_squared_error
from sklearn.metrics import r2_score
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
    x = scaler.transform(x)
    print("after",x.shape)

    xtrain, xtest, ytrain, ytest = train_test_split(x, y, test_size=0.3, random_state=420)

    rfc = RandomForestRegressor(n_estimators=100)
    rfc.fit(xtrain,ytrain)
    ypred = rfc.predict(xtrain)
    train_mes = mean_squared_error(ytrain, ypred)
    train_r2 = r2_score(ytrain, ypred)
    print("mean_squared_error", train_mes, train_r2)


    ypred=rfc.predict(xtest)
    test_mes=mean_squared_error(ytest,ypred)
    test_r2=r2_score(ytest,ypred)
    print("mean_squared_error",test_mes,test_r2)
    max_depth=[estimator.tree_.max_depth for estimator in rfc.estimators_[:30]]
    print("max_depth",max_depth)

    # min_samples_split=[estimator.tree_.min_samples_split for estimator in rfc.estimators_[:30]]
    # print("min_samples_split",min_samples_split)
    # min_samples_leaf = [estimator.tree_.min_samples_leaf for estimator in rfc.estimators_[:30]]
    # print("min_samples_leaf", min_samples_leaf)








if __name__=="__main__":
    t1=time()
    Train()
    print("cost",time()-t1)
