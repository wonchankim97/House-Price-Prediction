import pandas as pd
import numpy as np
from math import sqrt
from sklearn.model_selection import GridSearchCV
from sklearn.model_selection import cross_val_score, KFold
from sklearn.metrics import mean_squared_error, make_scorer

def get_param(model):
    cv = 5
    n_jobs = -1
    verbose = 1
    scoring = make_scorer(mean_squared_error, greater_is_better=False)
    
    class_name = type(model).__name__ 
    
    if class_name == 'ElasticNet':        
        params = {'alpha': [0.0004,0.0005,0.0006,0.0007,0.0008,0.0009],
                  'l1_ratio': [0.5, 0.6,0.7,0.8],
                  'max_iter': [10000]}
        return cv, n_jobs, verbose, scoring, params
    elif class_name == 'RandomForestRegressor':
        params = {'max_depth': [3, 4, 5],
                  'min_samples_leaf': [3, 4, 5],
                  'n_estimators': [700,1000,1460]}
        return cv, n_jobs, verbose, scoring, params
    elif class_name == 'Lasso':
        params = {'alpha': [0.0001, 0.0002, 0.0003, 0.0004, 0.0005, 0.0006, 0.0007, 0.0008],
                 "normalize": [False]}
        return cv, n_jobs, verbose, scoring, params
    elif class_name == 'Ridge':
        params = {'alpha': [1,2,3,4,5,6,7,8]}
        return cv, n_jobs, verbose, scoring, params
    elif class_name == 'SVR':
        params = {'gamma': [1e-06, 1e-07, 1e-08, 2e-08],
                 'C': [70000,80000],
                 'epsilon': [0.1, 0.2],
                 'kernel':['rbf']}
        return cv, n_jobs, verbose, scoring, params
    elif class_name == 'XGBRegressor':
        params = {'learning_rate': [0.01],
                 'n_estimators': [3500],
                 'max_depth': [3],
                 'min_child_weight': [0],
                 'gamma': [0],
                 'subsample': [0.6,0.7],
                 'colsample_bytree': [0.6,0.7],
                 'objective': ['reg:linear'],
                 'scale_pos_weight': [1],
                 'seed': [25, 30],
                 'reg_alpha': [0.00005,0.00007]}
        return cv, n_jobs, verbose, scoring, params
    elif class_name == 'LGBMRegressor':
        params = {'objective': ['regression'],
                  'num_leaves': [70,60,80],
                  'max_depth': [3,4,5],
                  'bagging_seed': [0,1],
                  'max_bin':[10,20,30],
                  'sub_feature':[0.4,0.5,0.6],
                  'matric':['binary_logloss'],
                  'num_iterations':[300,500]}
        return cv, n_jobs, verbose, scoring, params

def grid_search(train_X, train_Y, model):
    cv, n_jobs, verbose, scoring, params = get_param(model)
    
    kf = KFold(cv, shuffle=True, random_state=777).get_n_splits(train_X)
    grid_model = GridSearchCV(model, param_grid=params, scoring=scoring, cv=kf, verbose=verbose, n_jobs=n_jobs)
    grid_model.fit(train_X, train_Y)
    
    print("Model: {} Score: ({}) Best params: {}".format(type(model).__name__, sqrt(-grid_model.best_score_), grid_model.best_params_))
    print(grid_model.best_estimator_)
    
    return grid_model.best_estimator_