import pandas as pd
import numpy as np
import lightgbm as lgb
import shap
import matplotlib.pyplot as plt
from sklearn.metrics import precision_recall_curve
from sklearn.model_selection import train_test_split
import os


print("Загрузка данных и финальной модели...")
df = pd.read_csv('data/application_train_features.csv')
model = lgb.Booster(model_file='models/final_lgbm.txt')

# Чистка (как при обучении)
df['DAYS_EMPLOYED'] = df['DAYS_EMPLOYED'].replace(365243, np.nan)
drop_cols = ['SK_ID_CURR', 'FLAG_MOBIL', 'FLAG_CONT_MOBILE', 'FLAG_DOCUMENT_2', 'FLAG_DOCUMENT_4', 
             'FLAG_DOCUMENT_7', 'FLAG_DOCUMENT_9', 'FLAG_DOCUMENT_10', 'FLAG_DOCUMENT_11', 'FLAG_DOCUMENT_12',
             'FLAG_DOCUMENT_13', 'FLAG_DOCUMENT_14', 'FLAG_DOCUMENT_15', 'FLAG_DOCUMENT_16', 
             'FLAG_DOCUMENT_17', 'FLAG_DOCUMENT_18', 'FLAG_DOCUMENT_19', 'FLAG_DOCUMENT_20', 
             'FLAG_DOCUMENT_21', 'AMT_REQ_CREDIT_BUREAU_HOUR', 'AMT_REQ_CREDIT_BUREAU_DAY',
             'FLAG_EMP_PHONE', 'LIVE_REGION_NOT_WORK_REGION', 'FLAG_DOCUMENT_8', 'FLAG_DOCUMENT_6']

y = df['TARGET']
X = df.drop(columns=drop_cols + ['TARGET'])

# Берем только топ-100 фичей
with open('models/top100_features.txt', 'r') as f:
    top100 = [line.strip() for line in f.readlines()]
top100 = [c for c in top100 if c in X.columns]
X = X[top100]

cat_cols = X.select_dtypes(include='object').columns
for c in cat_cols:
    X[c] = X[c].astype('category')

# Берем всего 5000 клиентов для SHAP (этого достаточно для графика)
np.random.seed(42)
sample_idx = np.random.choice(X.shape[0], size=5000, replace=False)
X_sample = X.iloc[sample_idx]
y_sample = y.iloc[sample_idx]

# 1. SHAP-АНАЛИЗ
print("Расчет SHAP-значений (займет 1-2 минуты)...")
explainer = shap.TreeExplainer(model)
shap_values = explainer.shap_values(X_sample)

plt.figure(figsize=(10, 8))
shap.summary_plot(shap_values, X_sample, show=False, max_display=15)
plt.title("SHAP: Топ-15 факторов риска дефолта", fontsize=14)
plt.tight_layout()
plt.savefig('reports/shap_summary_final.png', dpi=150, bbox_inches='tight')
plt.close()
print("График SHAP сохранен в reports/shap_summary_final.png")

# 2. АНАЛИЗ ПОРОГА ОТСЕЧЕНИЯ
print("Анализ порога отсечения...")
y_pred_proba = model.predict(X_sample)
precisions, recalls, thresholds = precision_recall_curve(y_sample, y_pred_proba)
f1_scores = 2 * (precisions * recalls) / (precisions + recalls + 1e-9)
best_idx = np.argmax(f1_scores)
best_threshold = thresholds[best_idx]

plt.figure(figsize=(10, 6))
plt.plot(thresholds, precisions[:-1], 'b--', label='Precision (Точность)', linewidth=2)
plt.plot(thresholds, recalls[:-1], 'g-', label='Recall (Полнота)', linewidth=2)
plt.plot(thresholds, f1_scores[:-1], 'r-', label='F1-score', linewidth=2, alpha=0.7)
plt.axvline(x=best_threshold, color='black', linestyle=':', label=f'Оптимальный порог (F1) = {best_threshold:.2f}')

plt.title('Баланс Precision и Recall в зависимости от порога', fontsize=14)
plt.xlabel('Порог вероятности дефолта', fontsize=12)
plt.ylabel('Значение метрики', fontsize=12)
plt.legend(loc='best')
plt.grid(True, alpha=0.3)
plt.tight_layout()
plt.savefig('reports/threshold_analysis_final.png', dpi=150)
print(f"График порога сохранен. Оптимальный порог: {best_threshold:.2f}")

# Смотрим какую долю клиетов модель отправит на проверку (9.892)
''' 
y_pred_all = model.predict(X)
flagged_count = (y_pred_all > 0.19).sum()
total_count = len(y_pred_all)
flagged_pct = (flagged_count / total_count) * 100
print(total_count)
print(flagged_pct)
'''