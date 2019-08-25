import sys
sys.path.append('../')

import pandas as pd
import numpy as np
from mlxtend.regressor import StackingRegressor
from sklearn.ensemble import RandomForestRegressor
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
XGBoost = grid_search(train_X, train_Y, xgb.XGBRegressor())
#light_GBM = grid_search(train_X, train_Y, lgb.LGBMRegressor())

stacked_regression = StackingRegressor(
        #regressors=[elastic_net, lasso, ridge, random_forest, XGBoost, light_GBM],
        regressors=[elastic_net, lasso, ridge,support_vector_regressor,XGBoost],
        meta_regressor=support_vector_regressor
)

stacked_regression.fit(train_X, train_Y)

stacked = stacked_regression.predict(test_X)

ensembled = np.expm1((0.2 * elastic_net.predict(test_X)) +
                     (0.1 * lasso.predict(test_X)) +
                     (0.1 * ridge.predict(test_X)) +
                     #(0.05 * random_forest.predict(test_X)) +
                     (0.3 * support_vector_regressor.predict(test_X)) +
                     #(0.2 * XGBoost.predict(test_X)) +
                     #(0.2 * light_GBM.predict(test_X)) +
                     (0.3 * stacked))


print(stacked_regression.score(train_X, train_Y))

### for testing ###
'''
stacked_regression.fit(train_X, train_Y)

stacked = stacked_regression.predict(train_X)

ensembled = np.expm1((0.2 * elastic_net.predict(train_X)) +
                     (0.2 * lasso.predict(train_X)) +
                     (0.1 * ridge.predict(train_X)) +
                     #(0.05 * random_forest.predict(train_X)) +
                     (0.1 * support_vector_regressor.predict(train_X)) +
                     (0.2 * XGBoost.predict(train_X)) +
                     #(0.2 * light_GBM.predict(train_X)) +
                     (0.2 * stacked))

RMSE = np.mean((ensembled - test_Y)**2)**(1/2)
print('Score : ' + str(RMSE))

predict = pd.DataFrame({
    'Id':train_X.index,
    'SalePrice':ensembled
})
predict.to_csv('data/predict.csv', index=False)
'''
###################



"""
Export submission data
"""

#df = pd.DataFrame({"price": ensembled})
#q1 = df["price"].quantile(0.0042)
#q2 = df["price"].quantile(0.99)

#df["price"] = df["price"].apply(lambda x: x if x > q1 else x * 0.77)
#df["price"] = df["price"].apply(lambda x: x if x < q2 else x * 1.1)

submission = pd.DataFrame({
    'Id':test_X.index + (len(train_X_bf) - len(train_X) + 1),
    'SalePrice':ensembled
})
submission.to_csv('data/submission.csv', index=False)
