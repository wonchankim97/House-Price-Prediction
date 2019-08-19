import pandas as pd
import numpy as np
from math import sqrt
from sklearn.model_selection import GridSearchCV
from sklearn.model_selection import cross_val_score, KFold
from sklearn.metrics import mean_squared_error, make_scorer

def get_param(model):
    cv = 5
    n_jobs = 8
    verbose = 1
    scoring = make_scorer(mean_squared_error, greater_is_better=False)
    
    if type(model).__name__ == 'ElasticNet':        
        params = {'alpha': [0.0003, 0.0004, 0.0005],
                  'l1_ratio': [0.9, 0.95, 0.99, 1],
                  'max_iter': [10000]}
        return cv, n_jobs, verbose, scoring, params
    elif type(model).__name__ == 'RandomForestRegressor':
        params = {'max_depth': [3, 4, 5],
                  'min_samples_leaf': [3, 4, 5],
                  'n_estimators': [700,1000,1460]}
        return cv, n_jobs, verbose, scoring, params
    elif type(model).__name__ == 'Lasso':
        params = {'alpha': [0.0005],
                 "normalize": [False]}
        return cv, n_jobs, verbose, scoring, params
    elif type(model).__name__ == 'Ridge':
        params = {'alpha': [10, 11]}
        return cv, n_jobs, verbose, scoring, params
    elif type(model).__name__ == 'SVR':
        params = {'gamma': ['scale'],
                 'C': [10000],
                 'epsilon': [0.1]}
        return cv, n_jobs, verbose, scoring, params
    elif type(model).__name__ == 'XGBRegressor':
        params = {'learning_rate': [0.05],
                  'max_depth': [5],
                  'subsample': [0.9],
                  'colsample_bytree': [0.5],
                  'silent': [True],                  
                  'n_estimators':[3000],
                  'refit' : [True]}
        return cv, n_jobs, verbose, scoring, params
    elif type(model).__name__ == 'LGBMRegressor':
        params = {'objective':['regression'],
                  'num_leave' : [1],
                  'learning_rate' : [0.05],
                  'n_estimators':[3000],
                  'max_bin' : [80],                  
                  'refit':[True]}
        return cv, n_jobs, verbose, scoring, params

def grid_search(train_X, train_Y, model):
    cv, n_jobs, verbose, scoring, params = get_param(model)
    
    kf = KFold(cv, shuffle=True, random_state=777).get_n_splits(train_X)
    grid_model = GridSearchCV(model, param_grid=params, scoring=scoring, cv=kf, verbose=verbose, n_jobs=n_jobs)
    grid_model.fit(train_X, train_Y)
    
    print("Estimator: {} score: ({}) best params: {}".format(type(model).__name__, sqrt(-grid_model.best_score_), grid_model.best_params_))
    print(grid_model.best_estimator_)
    
    return grid_model.best_estimator_