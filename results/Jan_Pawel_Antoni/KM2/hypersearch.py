import optuna
import xgboost as xgb


def objective(trial: optuna.Trial):
    params = {'objective': 'binary:logistic',
              'eval_metric': 'auc',
              'verbosity': 0,
              'nthread': 5,
              "max_depth": trial.suggest_int('max_depth', 1, 12),
              "eta": trial.suggest_float('eta', 0.001, 0.99),
              "gamma": trial.suggest_float('gamma', 0, 10),
              "subsample": trial.suggest_float('subsample', 0, 1),
              "lambda": trial.suggest_float('lambda', 1, 5),
              "alpha": trial.suggest_float('alpha', 0, 5),
              "colsample_bytree": trial.suggest_float("colsample_bytree", 0, 1),
              "colsample_bylevel": trial.suggest_float("colsample_bylevel", 0, 1),
              "colsample_bynode": trial.suggest_float("colsample_bynode", 0, 1),
              }


    results = {}

    history = xgb.cv(params=params, dtrain=d_train,num_boost_round=400,
                            nfold=4, metrics='auc',early_stopping_rounds=100,
                            seed=1, verbose_eval=False,show_stdv=True)

    trial.set_user_attr('n_estimators', history.shape[0])
    
    auc_score = history.loc[history.shape[0]-1, 'test-auc-mean']
    std = history.loc[history.shape[0]-1, 'test-auc-std']

    return auc_score, std


if __name__ == "__main__":
    


    #cats = [col for col in X_train_final if col.startswith('authors') or col.startswith('institutions') or 
    #col.startswith('countries') or col.startswith('mag_') or col.startswith('jour') or col.startswith('type_')]

    d_train = xgb.DMatrix('dtrain.buff')
    d_test = xgb.DMatrix('dtest.buff')
    study = optuna.create_study(
        study_name="xgboost", storage="sqlite:///trials.db",
        load_if_exists=True, directions=["maximize", "minimize"])
    study.optimize(objective, n_trials=2000)