"""
Скоринг клиента: возвращает вероятность дефолта + бизнес-решение + объяснение.
"""
import pandas as pd
import numpy as np
import lightgbm as lgb
import shap
import os

class CreditRiskScorer:
    def __init__(self,
                 model_path='models/final_lgbm.txt',
                 features_path='models/top100_features.txt',
                 data_path='data/application_train_features.csv'):
        print("Инициализация скорера...")
        self.model = lgb.Booster(model_file=model_path)
        with open(features_path) as f:
            self.features = [line.strip() for line in f]
        self.df = pd.read_csv(data_path)
        self.df['DAYS_EMPLOYED'] = self.df['DAYS_EMPLOYED'].replace(365243, np.nan)
        self.explainer = shap.TreeExplainer(self.model)
        print(f"Готов. Загружено {len(self.df)} клиентов, {len(self.features)} фичей в модели.")

    def _prepare(self, row: pd.DataFrame) -> pd.DataFrame:
        X = row.copy()
        # Категориальные -> category (как при обучении)
        cat_cols = X.select_dtypes(include='object').columns
        for c in cat_cols:
            X[c] = X[c].astype('category')
        # Оставляем только те фичи, что видела модель
        X = X.reindex(columns=self.features)
        return X

    def score(self, sk_id_curr: int,
              threshold_approve: float = 0.10,
              threshold_reject: float = 0.30,
              top_n: int = 5) -> dict:
        row = self.df[self.df['SK_ID_CURR'] == sk_id_curr]
        if row.empty:
            return {'error': f'Клиент {sk_id_curr} не найден'}

        X = self._prepare(row)
        proba = float(self.model.predict(X)[0])

        # Бизнес-решение по порогам
        if proba < threshold_approve:
            decision = 'APPROVE'
        elif proba < threshold_reject:
            decision = 'MANUAL_REVIEW'
        else:
            decision = 'REJECT'

        # SHAP-объяснение для ЭТОГО клиента
        shap_vals = self.explainer.shap_values(X)
        # Для бинарной классификации SHAP возвращает список [class_0, class_1]
        sv = shap_vals[1][0] if isinstance(shap_vals, list) else shap_vals[0]
        top_idx = np.argsort(np.abs(sv))[::-1][:top_n]

        top_factors = [
            {
                'feature': self.features[i],
                'value': X.iloc[0, i] if pd.notna(X.iloc[0, i]) else None,
                'shap_impact': round(float(sv[i]), 4),
                'direction': '↑ риск' if sv[i] > 0 else '↓ риск'
            }
            for i in top_idx
        ]

        return {
            'sk_id_curr': int(sk_id_curr),
            'default_probability': round(proba, 4),
            'decision': decision,
            'top_risk_factors': top_factors
        }


# --- Демо: скорим 5 случайных клиентов ---
if __name__ == '__main__':
    scorer = CreditRiskScorer()
    sample_ids = scorer.df['SK_ID_CURR'].sample(5, random_state=42).tolist()

    for cid in sample_ids:
        result = scorer.score(cid)
        print("\n" + "=" * 60)
        print(f"Клиент: {result['sk_id_curr']}")
        print(f"Вероятность дефолта: {result['default_probability']:.2%}")
        print(f"Решение: {result['decision']}")
        print("Топ-5 факторов:")
        for f in result['top_risk_factors']:
            print(f"  • {f['feature']:<35} = {f['value']!s:<12} "
                  f"({f['direction']}, SHAP={f['shap_impact']:+.3f})")