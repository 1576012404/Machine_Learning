# coding: utf-8
import lightgbm as lgb
import pandas as pd

from sklearn.metrics import mean_squared_error,r2_score
import numpy as np
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score
import pandas as pd

from sklearn.preprocessing import StandardScaler
from sklearn.feature_selection import VarianceThreshold
from time import time


def Train():
    filename = "train_reg.csv"
    data = pd.read_csv(filename, index_col=0, nrows=10)
    float_cols = [c for c in data if data[c].dtype == "float64"]
    float32_cols = {c: np.float32 for c in float_cols}
    data = pd.read_csv(filename, index_col=0, dtype=float32_cols)
    x = data.values[:, :-1]
    y = data.label
    print("pre", x.shape)
    scaler = VarianceThreshold()
    scaler.fit(x)
    x = scaler.transform(x)
    print("after", x.shape)

    xtrain, xtest, ytrain, ytest = train_test_split(x, y, test_size=0.3, random_state=420)

    # create dataset for lightgbm
    lgb_train = lgb.Dataset(xtrain, ytrain)
    lgb_eval = lgb.Dataset(xtest, ytest, reference=lgb_train)

    # specify your configurations as a dict
    # params = {
    #     'boosting_type': 'gbdt',
    #     'objective': 'regression',
    #     'metric': {'l2', 'l1'},
    #     # 'num_leaves': 31,
    #     # 'learning_rate': 0.05,
    #     # 'feature_fraction': 0.9,
    #     'bagging_fraction': 0.8,
    #     'bagging_freq': 5,
    #     'verbose': 0
    # }
    params = {
        'learning_rate': 0.3,
        'num_iterations': 500,
        'max_depth': 5,
        'n_estimators': 2500,
        'metric': {'l2', 'l1'},
    }

    print('Starting training...')
    # train
    gbm = lgb.train(params,
                    lgb_train,
                    num_boost_round=20,
                    valid_sets=lgb_eval,
                    early_stopping_rounds=5)

    print('Saving model...')
    # save model to file
    gbm.save_model('model.txt')

    print('Starting predicting...')
    # predict
    train_pred=gbm.predict(xtrain, num_iteration=gbm.best_iteration)
    print('train mse:', mean_squared_error(ytrain, train_pred), r2_score(ytrain, train_pred))

    y_pred = gbm.predict(xtest, num_iteration=gbm.best_iteration)
    # eval
    print('test mse:', mean_squared_error(ytest, y_pred),r2_score(ytest,y_pred))


if __name__=="__main__":
    t1=time()
    Train()
    print("cost",time()-t1)