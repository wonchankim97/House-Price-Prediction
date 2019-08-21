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
        params = {'alpha': [0.0007],
                  'l1_ratio': [0.7],
                  'max_iter': [10000]}
        return cv, n_jobs, verbose, scoring, params
    elif class_name == 'RandomForestRegressor':
        params = {'max_depth': [3, 4, 5],
                  'min_samples_leaf': [3, 4, 5],
                  'n_estimators': [700,1000,1460]}
        return cv, n_jobs, verbose, scoring, params
    elif class_name == 'Lasso':
        params = {'alpha': [0.0005],
                 "normalize": [False]}
        return cv, n_jobs, verbose, scoring, params
    elif class_name == 'Ridge':
        params = {'alpha': [10, 11]}
        return cv, n_jobs, verbose, scoring, params
    elif class_name == 'SVR':
        params = {'gamma': ['scale'],
                 'C': [20000],
                 'epsilon': [0.0001, 0.001, 0.01]}
        return cv, n_jobs, verbose, scoring, params
    elif class_name == 'XGBRegressor':
        params = {'learning_rate': [0.01],
                  'min_child_weight':[3],
                  'max_depth': [3],
                  'gamma':[0],
                  'subsample': [0.7],
                  'colsample_bytree': [0.6],
                  'silent': [True],                  
                  'n_estimators':[4000],
                  'refit' : [True]}        
        return cv, n_jobs, verbose, scoring, params
    elif class_name == 'LGBMRegressor':
        params = {'objective':['regression'],
                  'num_leave' : [1],
                  'learning_rate' : [0.01],
                  'n_estimators':[3500],
                  'max_bin' : [800],                  
                  'refit':[True]}
        return cv, n_jobs, verbose, scoring, params

def grid_search(train_X, train_Y, model):
    cv, n_jobs, verbose, scoring, params = get_param(model)
    
    kf = KFold(cv, shuffle=True, random_state=777).get_n_splits(train_X)
    grid_model = GridSearchCV(model, param_grid=params, scoring=scoring, cv=kf, verbose=verbose, n_jobs=n_jobs)
    grid_model.fit(train_X, train_Y)
    
    print("Model: {} Score: ({}) Best params: {}".format(type(model).__name__, sqrt(-grid_model.best_score_), grid_model.best_params_))
    print(grid_model.best_estimator_)
    
    return grid_model.best_estimator_