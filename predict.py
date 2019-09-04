import sys
sys.path.append('../')

import pandas as pd
import numpy as np
from mlxtend.regressor import StackingRegressor
from sklearn.ensemble import RandomForestRegressor, GradientBoostingRegressor
from sklearn.linear_model import Lasso, Ridge, ElasticNet
from sklearn.svm import SVR
import xgboost as xgb
import lightgbm as lgb

from modules.featureEng import *
from modules.modelEng import *

# to display entire data
#pd.set_option('display.max_rows', 1500)
#pd.set_option('display.max_columns', 150)

"""
Load Data
"""

train_df = pd.read_csv('data/train.csv', index_col='Id')
test_df = pd.read_csv('data/test.csv', index_col='Id')

"""
Feature Engineering
"""

train_X_bf = train_df.loc[:,train_df.columns!='SalePrice']
train_Y_bf = train_df['SalePrice']
test_X_bf = test_df

### for testing ###
#from sklearn.model_selection import train_test_split
#train_X_bf, test_X_bf, train_Y_bf, test_Y = train_test_split(train_X_bf, train_Y_bf, random_state=777)
###################

train_X, train_Y, test_X = pre_processing(train_X_bf.copy(), train_Y_bf.copy(), test_X_bf.copy())

"""
Predict house price
"""

elastic_net = grid_search(train_X, train_Y, ElasticNet())
lasso = grid_search(train_X, train_Y, Lasso())
ridge = grid_search(train_X, train_Y, Ridge())
#random_forest = grid_search(train_X, train_Y, RandomForestRegressor())
support_vector_regressor = grid_search(train_X, train_Y, SVR())
#gradient_boost_regressor = grid_search(train_X, train_Y, GradientBoostingRegressor())
XGBoost = grid_search(train_X, train_Y, xgb.XGBRegressor())
light_GBM = grid_search(train_X, train_Y, lgb.LGBMRegressor())

stacked_regression = StackingRegressor(
        regressors=[elastic_net, lasso, ridge, support_vector_regressor, XGBoost, light_GBM],
        meta_regressor=support_vector_regressor
)

stacked_regression.fit(train_X, train_Y)

stacked = stacked_regression.predict(test_X)

ensembled = np.expm1((0.1 * elastic_net.predict(test_X)) +
                     (0.2 * lasso.predict(test_X)) +
                     (0.1 * ridge.predict(test_X)) +
                     (0.1 * support_vector_regressor.predict(test_X)) +
                     (0.2 * XGBoost.predict(test_X)) +
                     (0.1 * light_GBM.predict(test_X)) +
                     (0.2 * stacked))


print(stacked_regression.score(train_X, train_Y))

"""
Export submission data
"""
submission = pd.DataFrame({
    'Id':test_X.index + (len(train_X_bf) - len(train_X) + 1),
    'SalePrice':ensembled
})
submission.to_csv('data/submission.csv', index=False)
