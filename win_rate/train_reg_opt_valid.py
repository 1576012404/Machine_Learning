from hyperopt import hp
import numpy as np
from sklearn.metrics import mean_squared_error
import lightgbm as lgb
import pandas as pd
from sklearn.feature_selection import VarianceThreshold
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split
from hyperopt import fmin, tpe, STATUS_OK, STATUS_FAIL, Trials
#lightGBM parameters


filename = "train_classify.csv"
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

stdScaler=StandardScaler()
stdScaler.fit(x)
x = stdScaler.transform(x)

print("after", x.shape)

xtrain, xtest, ytrain, ytest = train_test_split(x, y, test_size=0.3, random_state=420)


lgb_reg_params = {
    'learning_rate':      0.2,
    'max_depth':         10,
    # 'min_child_weight': hp.choice('min_child_weight', np.arange(1, 8, 1, dtype=int)),
    # 'colsample_bytree': hp.choice('colsample_bytree', np.arange(0.3, 0.8, 0.1)),
    # 'subsample':        hp.uniform('subsample', 0.8, 1),
    'n_estimators':     100,
}
lgb_fit_params = {
    'eval_metric': 'l2',
    'early_stopping_rounds': 10,
    'verbose': False
}


reg = lgb.LGBMRegressor(**lgb_reg_params)
reg.fit(xtrain, ytrain,
                eval_set=[(xtrain, ytrain), (xtest,ytest)],
                **lgb_fit_params)
pred = reg.predict(xtest)

mse=mean_squared_error(ytest,pred)
print("mse",mse)#0.0011090987221364216