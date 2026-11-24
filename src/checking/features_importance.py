import pandas as pd
import numpy as np
import lightgbm as lgb
import matplotlib.pyplot as plt
import os


# 1. Загружаем обученную модель
print("Загрузка модели...")
model = lgb.Booster(model_file='models/lgbm_with_features.txt')
feature_names = model.feature_name()

# 2. Получаем важность по gain (сколько "пользы" фича принесла при разбиениях)
importance_gain = model.feature_importance(importance_type='gain')
importance_split = model.feature_importance(importance_type='split')

imp_df = pd.DataFrame({
    'feature': feature_names,
    'gain': importance_gain,
    'split': importance_split
}).sort_values('gain', ascending=False).reset_index(drop=True)

print(f"\nВсего фичей: {len(imp_df)}")

# 3. Сколько фичей с нулевой важностью (модель их вообще не использовала)
zero_imp = imp_df[imp_df['gain'] == 0]
print(f"Фичей с нулевой важностью (gain=0): {len(zero_imp)} ({len(zero_imp)/len(imp_df):.1%})")
if len(zero_imp) > 0:
    print("Список нулевых фичей:")
    print(zero_imp['feature'].tolist())

# 4. Сколько фичей дают 95% суммарной важности (принцип Парето)
imp_df['gain_cumsum'] = imp_df['gain'].cumsum() / imp_df['gain'].sum()
n_95 = (imp_df['gain_cumsum'] <= 0.95).sum() + 1
print(f"\nФичей, дающих 95% суммарной важности: {n_95} из {len(imp_df)}")
print(f"Фичей, дающих 99% суммарной важности: {(imp_df['gain_cumsum'] <= 0.99).sum() + 1}")

# 5. Топ-30 фичей
print("\nТоп-30 фичей по важности (gain):")
print(imp_df[['feature', 'gain', 'split']].head(30).to_string(index=False))

# 6. Разбиваем на группы: исходные vs bureau vs prev vs installments
def group_of(fname):
    if fname.startswith('BUREAU_'): return 'BUREAU'
    if fname.startswith('PREV_'): return 'PREV'
    if fname.startswith('INST_'): return 'INSTALLMENTS'
    return 'APPLICATION (исходные)'

imp_df['group'] = imp_df['feature'].apply(group_of)
group_stats = imp_df.groupby('group').agg(
    n_features=('feature', 'count'),
    total_gain=('gain', 'sum')
).sort_values('total_gain', ascending=False)
group_stats['gain_share'] = (group_stats['total_gain'] / group_stats['total_gain'].sum()).round(3)

print("\nВклад групп фичей в суммарную важность:")
print(group_stats.to_string())

# 7. График топ-30
plt.figure(figsize=(10, 10))
top30 = imp_df.head(30).iloc[::-1]  # разворачиваем, чтобы топ был сверху
plt.barh(top30['feature'], top30['gain'], color='steelblue', edgecolor='black')
plt.xlabel('Важность (gain)', fontsize=12)
plt.title('Топ-30 фичей по важности (gain)', fontsize=14)
plt.tight_layout()
plt.savefig('reports/feature_importance_top30.png', dpi=150, bbox_inches='tight')
print("\nГрафик сохранен в reports/feature_importance_top30.png")

# 8. Сохраняем список топ-100 для последующего отбора
top100 = imp_df.head(100)['feature'].tolist()
with open('models/top100_features.txt', 'w') as f:
    f.write('\n'.join(top100))
print(f"Список топ-100 фичей сохранен в models/top100_features.txt")