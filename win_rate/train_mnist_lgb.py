
from sklearn import datasets, svm, metrics
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score
import numpy as np

import time
digits = datasets.load_digits()
print("digit",digits.images.shape)
n_samples = len(digits.images)
data = digits.images.reshape((n_samples, -1))
X_train, X_test, y_train, y_test = train_test_split(
    data, digits.target, test_size=0.5, shuffle=True)


import lightgbm as lgb
train_data = lgb.Dataset(X_train, label=y_train)
test_data = lgb.Dataset(X_test, label=y_test)

# specify parameters via map
# params = {
#     'num_leaves':31,                # Same to max_leaf_nodes in GBDT, but GBDT's default value is None
#     'max_depth': -1,                # Same to max_depth of xgboost
#     'tree_learner': 'serial',
#     'application':'multiclass',     # Same to objective of xgboost
#     'num_class':10,                 # Same to num_class of xgboost
#     'learning_rate': 0.1,           # Same to eta of xgboost
#     'min_split_gain': 0,            # Same to gamma of xgboost
#     'lambda_l1': 0,                 # Same to alpha of xgboost
#     'lambda_l2': 0,                 # Same to lambda of xgboost
#     'min_data_in_leaf': 20,         # Same to min_samples_leaf of GBDT
#     'bagging_fraction': 1.0,        # Same to subsample of xgboost
#     'bagging_freq': 0,
#     'bagging_seed': 0,
#     'feature_fraction': 1.0,         # Same to colsample_bytree of xgboost
#     'feature_fraction_seed': 2,
#     'min_sum_hessian_in_leaf': 1e-3, # Same to min_child_weight of xgboost
#     'num_threads': 1
# }

params=dict(
    task="train",
    application="binary",
    num_class=10,
    boosting="gbdt",
    objective="multiclass",
    metric="multi_logloss",
    metric_freq=50,
    is_training_metrics=False,
    max_depth=4,
    num_leaves=31,
    learning_rate=0.1,
    feature_fraction=1.0,
    bagging_fraction=1.0,
    bagging_freq=0,
    bagging_seed=2018,
    verbose=0,
    num_thread=16,
)
num_round = 2000

# start training
start_time = time.time()
bst = lgb.train(params, train_data, num_round)
end_time = time.time()
print('The training time = {}'.format(end_time - start_time))

# get prediction and evaluate
ypred_onehot = bst.predict(X_test)
ypred = []
for i in range(len(ypred_onehot)):
    ypred.append(ypred_onehot[i].argmax())

accuracy = np.sum(ypred == y_test) / len(ypred)
print('Test accuracy = {}'.format(accuracy))



# predicted = classifier.predict(X_test)
# acc=accuracy_score(y_test,predicted)
# print("acc",acc)#0.9688