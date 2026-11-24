import pandas as pd
import numpy as np
import lightgbm as lgb
from sklearn.model_selection import StratifiedKFold
from sklearn.metrics import roc_auc_score
import optuna
import json

optuna.logging.set_verbosity(optuna.logging.WARNING)

print("Загрузка данных...")
df = pd.read_csv('data/application_train_features.csv')
df['DAYS_EMPLOYED'] = df['DAYS_EMPLOYED'].replace(365243, np.nan)

drop_cols = ['SK_ID_CURR', 'FLAG_MOBIL', 'FLAG_CONT_MOBILE', 'FLAG_DOCUMENT_2', 'FLAG_DOCUMENT_4', 
             'FLAG_DOCUMENT_7', 'FLAG_DOCUMENT_9', 'FLAG_DOCUMENT_10', 'FLAG_DOCUMENT_11', 'FLAG_DOCUMENT_12',
             'FLAG_DOCUMENT_13', 'FLAG_DOCUMENT_14', 'FLAG_DOCUMENT_15', 'FLAG_DOCUMENT_16', 
             'FLAG_DOCUMENT_17', 'FLAG_DOCUMENT_18', 'FLAG_DOCUMENT_19', 'FLAG_DOCUMENT_20', 
             'FLAG_DOCUMENT_21', 'AMT_REQ_CREDIT_BUREAU_HOUR', 'AMT_REQ_CREDIT_BUREAU_DAY',
             'FLAG_EMP_PHONE', 'LIVE_REGION_NOT_WORK_REGION', 'FLAG_DOCUMENT_8', 'FLAG_DOCUMENT_6']

y = df['TARGET']
X = df.drop(columns=drop_cols + ['TARGET'])

# Оставляем только топ-100
with open('models/top100_features.txt', 'r') as f:
    top100 = [line.strip() for line in f.readlines()]
top100 = [c for c in top100 if c in X.columns]
X = X[top100]

cat_cols = X.select_dtypes(include='object').columns
for c in cat_cols:
    X[c] = X[c].astype('category')

def objective(trial):
    params = {
        'objective': 'binary',
        'metric': 'auc',
        'boosting_type': 'gbdt',
        'random_state': 42,
        'verbose': -1,
        'n_jobs': -1,
        # Параметры для поиска
        'learning_rate': trial.suggest_float('learning_rate', 0.01, 0.1, log=True),
        'num_leaves': trial.suggest_int('num_leaves', 20, 100),
        'max_depth': trial.suggest_int('max_depth', 3, 12),
        'min_child_samples': trial.suggest_int('min_child_samples', 20, 200),
        'feature_fraction': trial.suggest_float('feature_fraction', 0.6, 1.0),
        'subsample': trial.suggest_float('subsample', 0.7, 1.0),
        'reg_alpha': trial.suggest_float('reg_alpha', 1e-8, 10.0, log=True),
        'reg_lambda': trial.suggest_float('reg_lambda', 1e-8, 10.0, log=True),
    }

    skf = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)
    aucs = []
    
    for tr, va in skf.split(X, y):
        dtrain = lgb.Dataset(X.iloc[tr], y.iloc[tr])
        dval = lgb.Dataset(X.iloc[va], y.iloc[va])
        
        model = lgb.train(
            params, dtrain, num_boost_round=2000,
            valid_sets=[dval],
            callbacks=[lgb.early_stopping(50, verbose=False)]
        )
        pred = model.predict(X.iloc[va])
        aucs.append(roc_auc_score(y.iloc[va], pred))
        
    return np.mean(aucs)

print("Запуск Optuna (20 trials)... Это займет около 15 минут.")
study = optuna.create_study(direction='maximize')
study.optimize(objective, n_trials=20, show_progress_bar=True)

print("\nЛучшие параметры:")
print(study.best_params)
print(f"Лучший CV AUC: {study.best_value:.5f}")

with open('models/best_params.json', 'w') as f:
    json.dump(study.best_params, f)
print("Параметры сохранены в models/best_params.json")