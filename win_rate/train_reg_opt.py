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
    'learning_rate':    hp.choice('learning_rate',    np.arange(0.05, 0.31, 0.05)),
    'max_depth':        hp.choice('max_depth',        np.arange(5, 16, 5, dtype=int)),
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
lgb_para = dict()
lgb_para['reg_params'] = lgb_reg_params
lgb_para['fit_params'] = lgb_fit_params
lgb_para['loss_func' ] = lambda y, pred: mean_squared_error(y, pred)


class HPOpt(object):
    def __init__(self, x_train, x_test, y_train, y_test):
        self.x_train = x_train
        self.x_test  = x_test
        self.y_train = y_train
        self.y_test  = y_test

    def process(self, fn_name, space, trials, algo, max_evals):
        fn = getattr(self, fn_name)
        try:
            result = fmin(fn=fn, space=space, algo=algo, max_evals=max_evals, trials=trials)
        except Exception as e:
            return {'status': STATUS_FAIL,
                    'exception': str(e)}
        return result, trials

    def lgb_reg(self, para):
        reg = lgb.LGBMRegressor(**para['reg_params'])
        return self.train_reg(reg, para)


    def train_reg(self, reg, para):
        reg.fit(self.x_train, self.y_train,
                eval_set=[(self.x_train, self.y_train), (self.x_test, self.y_test)],
                **para['fit_params'])
        pred = reg.predict(self.x_test)
        loss = para['loss_func'](self.y_test, pred)
        return {'loss': loss, 'status': STATUS_OK}



obj = HPOpt(xtrain, xtest, ytrain, ytest)

result, trials = obj.process(fn_name='lgb_reg', space=lgb_para, trials=Trials(), algo=tpe.suggest, max_evals=100)
print("result",result)
print("trials",trials)

