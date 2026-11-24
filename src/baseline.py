import pandas as pd
import numpy as np
import lightgbm as lgb
from sklearn.model_selection import StratifiedKFold
from sklearn.metrics import roc_auc_score
import os

os.makedirs('models', exist_ok=True)

# 1. Загрузка
print("Загрузка данных...")
df = pd.read_csv('data/application_train.csv')

# 2. Базовая чистка (по итогам EDA)
# Убираем аномалию
df['DAYS_EMPLOYED'] = df['DAYS_EMPLOYED'].replace(365243, np.nan)

# Список колонок к удалению (ID и почти-константы из EDA)
drop_cols = ['SK_ID_CURR', 'FLAG_MOBIL', 'FLAG_CONT_MOBILE',
             'FLAG_DOCUMENT_2','FLAG_DOCUMENT_4','FLAG_DOCUMENT_7','FLAG_DOCUMENT_9',
             'FLAG_DOCUMENT_10','FLAG_DOCUMENT_11','FLAG_DOCUMENT_12','FLAG_DOCUMENT_13',
             'FLAG_DOCUMENT_14','FLAG_DOCUMENT_15','FLAG_DOCUMENT_16','FLAG_DOCUMENT_17',
             'FLAG_DOCUMENT_18','FLAG_DOCUMENT_19','FLAG_DOCUMENT_20','FLAG_DOCUMENT_21',
             'AMT_REQ_CREDIT_BUREAU_HOUR','AMT_REQ_CREDIT_BUREAU_DAY']

y = df['TARGET']
X = df.drop(columns=drop_cols + ['TARGET'])

# 3. Категориальные фичи -> тип category (чтобы LightGBM понял)
cat_cols = X.select_dtypes(include='object').columns
for c in cat_cols:
    X[c] = X[c].astype('category')

# 4. Параметры модели
params = {
    'objective': 'binary',
    'metric': 'auc',
    'boosting_type': 'gbdt',
    'num_leaves': 31,
    'learning_rate': 0.05,
    'feature_fraction': 0.8,
    'random_state': 42,
    'verbose': -1,
    'n_jobs': -1
}

# 5. Кросс-валидация
skf = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)
aucs = []
models = []

print("Начинаем кросс-валидацию (5 фолдов)...")
for fold, (tr, va) in enumerate(skf.split(X, y), 1):
    dtrain = lgb.Dataset(X.iloc[tr], y.iloc[tr])
    dval   = lgb.Dataset(X.iloc[va], y.iloc[va])
    
    model = lgb.train(
        params, 
        dtrain, 
        num_boost_round=2000,
        valid_sets=[dval],
        callbacks=[lgb.early_stopping(100, verbose=False), lgb.log_evaluation(0)]
    )
    
    pred = model.predict(X.iloc[va])
    auc = roc_auc_score(y.iloc[va], pred)
    aucs.append(auc)
    models.append(model)
    print(f"Fold {fold}: AUC = {auc:.5f}, лучшая итерация: {model.best_iteration}")

print(f"\nCV AUC: {np.mean(aucs):.5f} ± {np.std(aucs):.5f}")

# Сохраняем модель первого фолда (для инференса потом)
models[0].save_model('models/baseline_lgbm.txt')
print("Модель сохранена в models/baseline_lgbm.txt")