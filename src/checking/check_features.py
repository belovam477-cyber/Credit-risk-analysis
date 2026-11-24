import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import os


df = pd.read_csv('data/application_train_features.csv')
print(f"Размер датасета: {df.shape}")


# 1. ПРОВЕРКА НА INF И ПРОПУСКИ В НОВЫХ ФИЧАХ
print("\n--- 1. Проверка новых фич на inf и пропуски ---")

new_cols = [c for c in df.columns if c.startswith('BUREAU_') or c.startswith('PREV_')]
print(f"Всего новых фич: {len(new_cols)}")

# 1.1 Проверка на бесконечности (inf)
inf_counts = np.isinf(df[new_cols]).sum()
if inf_counts.sum() > 0:
    print("ВНИМАНИЕ! Найдены бесконечности (inf) в колонках:")
    print(inf_counts[inf_counts > 0])
else:
    print("Проверка на inf: пройдена (бесконечностей нет).")

# 1.2 Проверка пропусков
missing_new = df[new_cols].isnull().mean().sort_values(ascending=False)
print(f"\nСредняя доля пропусков в новых фичах: {missing_new.mean():.2%}")
print("Топ-5 новых фич с наибольшим числом пропусков:")
print(missing_new.head(5).round(3))


# 2. ПОИСК УТЕЧЕК (LEAKAGE) И КОРРЕЛЯЦИИ С TARGET
print("\n--- 2. Корреляция новых фич с TARGET ---")
# Берем только новые фичи и TARGET
corr_new = df[new_cols + ['TARGET']].corr()['TARGET'].drop('TARGET')
corr_new_abs = corr_new.abs().sort_values(ascending=False)

print("Топ-10 новых фич по корреляции с TARGET:")
print(corr_new_abs.head(10).round(4))

# Если какая-то фича коррелирует с TARGET > 0.8 — это подозрительно (утечка)
leaks = corr_new_abs[corr_new_abs > 0.8]
if not leaks.empty:
    print("\nВНИМАНИЕ! Возможная утечка данных (корреляция > 0.8):")
    print(leaks)
else:
    print("\nПроверка на утечки: пройдена (нет фич с корреляцией > 0.8).")

# Строим график топ-15 корреляций
plt.figure(figsize=(10, 6))
corr_new_abs.head(20).plot(kind='barh', color='teal', edgecolor='black')
plt.title('Топ-15 новых фич по корреляции с TARGET', fontsize=14)
plt.xlabel('Абсолютная корреляция', fontsize=12)
plt.gca().invert_yaxis() # Чтобы самая важная фича была сверху
plt.tight_layout()
plt.savefig('reports/new_features_correlation.png', dpi=150)


# 3. ПРОВЕРКА ДУБЛИКАТОВ И МУЛЬТИКОЛЛИНЕАРНОСТИ
print("\n--- 3. Поиск сильно коррелирующих пар (мультиколлинеарность) ---")
# Считаем матрицу корреляций только для новых фич
corr_matrix = df[new_cols].corr().abs()

# Оставляем только верхний треугольник матрицы (чтобы не считать пары дважды)
upper = corr_matrix.where(np.triu(np.ones(corr_matrix.shape), k=1).astype(bool))

# Ищем пары с корреляцией > 0.95 (почти дубликаты)
high_corr_pairs = [(col, row, upper.loc[row, col]) 
                   for col in upper.columns 
                   for row in upper.index 
                   if upper.loc[row, col] > 0.95]

print(f"Найдено пар с корреляцией > 0.95: {len(high_corr_pairs)}")
if high_corr_pairs:
    print("Примеры таких пар (возможные дубликаты):")
    for c1, c2, val in high_corr_pairs[:5]:
        print(f"  {c1} <--> {c2}: {val:.3f}")