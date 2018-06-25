def score(params):
    num_round = int(params['n_estimators'])
    del params['n_estimators']
    dtrain = xgboost.DMatrix(train_x, label=train_y)
    dvalid = xgboost.DMatrix(test_x, label=test_y)
    watchlist = [(dvalid, 'eval'), (dtrain, 'train')]
    gbm_model = xgboost.train(params, 
                              dtrain, 
                              num_round,
                              evals=watchlist,
                              verbose_eval=False)
    predictions = gbm_model.predict(dvalid, ntree_limit=gbm_model.best_iteration)
    print(roc_auc_score(test_y, np.array(predictions)))
    loss = 1 - roc_auc_score(test_y, np.array(predictions))
    return {'loss': loss, 'status': STATUS_OK}
 
 
def optimize(evals, cores, trials, optimizer=tpe.suggest, random_state=0):
    space = {
        'n_estimators': hp.quniform('n_estimators', 200, 600, 1),
        'eta': hp.quniform('eta', 0.025, 0.25, 0.025), # A problem with max_depth casted to float instead of int with the hp.quniform method.
        'max_depth':  hp.choice('max_depth', np.arange(1, 14, dtype=int)),
        'min_child_weight': hp.quniform('min_child_weight', 1, 10, 1),
        'subsample': hp.quniform('subsample', 0.7, 1, 0.05),
        'gamma': hp.quniform('gamma', 0.5, 1, 0.05),
        'colsample_bytree': hp.quniform('colsample_bytree', 0.7, 1, 0.05),
        'alpha' :  hp.quniform('alpha', 0, 10, 1),
        'lambda': hp.quniform('lambda', 1, 2, 0.1),
        'nthread': cores,
        'objective': 'binary:logistic',
        'booster': 'gbtree',
        'seed': random_state
    }
    best = fmin(score, space, algo=tpe.suggest, max_evals=evals, trials = trials)
    return best
    

trials = Trials()
cores = 32
n= 1000
start = time.time()
best_param = optimize(evals = n,
                      optimizer=tpe.suggest,
                      cores = cores,
                      trials = trials)
print("------------------------------------")
print("The best hyperparameters are: ", "\n")
print(best_param)
end = time.time()
print('Time elapsed to optimize {0} executions: {1}'.format(n,end - start))



# Predicting with the best parameters obtained with HyperOpt:
dtrain = xgboost.DMatrix(train_x, label=train_y)
dvalid = xgboost.DMatrix(test_x, label=test_y)
watchlist = [(dvalid, 'eval'),(dtrain, 'train')]
gbm_model = xgboost.train(params={'alpha': 5.0, 'booster': 'gbtree', 'colsample_bytree': 0.9500000000000001, 'eta': 0.05, 'gamma': 0.75, 'lambda': 1.2000000000000002, 'max_depth': 2, 'min_child_weight': 8.0, 'nthread': 32, 'objective': 'binary:logistic', 'seed': 0, 'subsample': 0.9, 'eval_metric':'auc'}, 
                              dtrain=dtrain, 
                              num_boost_round=int(best_param['n_estimators']),
                              early_stopping_rounds=20,
                              evals=watchlist,
                              verbose_eval=True)
predictions = gbm_model.predict(dvalid, ntree_limit=gbm_model.best_iteration+1)

print(roc_auc_score(test_y,predictions))
