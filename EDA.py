import pandas as pd
import numpy as np
import matplotlib
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.feature_selection import mutual_info_classif

# Оценка формы, типа и объема данных
df = pd.read_csv('./data/application_train.csv')
print(df.shape)
print(df.dtypes.value_counts())
print(df.head(10))

# Дисбаланс классов
print(df['TARGET'].value_counts(normalize=True).round(4))
df['TARGET'].value_counts().plot(kind='bar', title='Class balance')
plt.show()

# Пропуски 
missing = df.isnull().mean().sort_values(ascending=False)
print(missing.head(20))
missing.hist(bins=30)
plt.show()

# Провека аномалий и заглушек
num_cols = df.select_dtypes(include=np.number).columns
print(df[num_cols].describe().T.sort_values('max', ascending=False).head(15))
#print((df['DAYS_EMPLOYED'] == 365243).mean())

# Поиск констант и идентификаторов(ID)
constant_cols = [c for c in df.columns if df[c].nunique() <= 1]
near_const   = [c for c in df.columns if df[c].value_counts(normalize=True).iloc[0] > 0.99]
print("Константы:", constant_cols)
print("Почти константы:", near_const)

# Категориальные колонки
cat_cols = df.select_dtypes(include='object').columns
print("Категориальные колонки: ", cat_cols)
print(df[cat_cols].nunique().sort_values(ascending=False))

# Mutual information ( сигналы в числовых значениях)
sample = df.sample(n=20000, random_state=42)
X = sample.drop(columns=['TARGET','SK_ID_CURR']).select_dtypes(include=np.number).fillna(-999)
mi = pd.Series(
    mutual_info_classif(X, sample['TARGET'], random_state=42, n_neighbors=2),
    index=X.columns
).sort_values(ascending=False)
print(mi.head(25))

# Корреляции 
corr = df[num_cols].corr()['TARGET'].abs().sort_values(ascending=False)
print(corr.head(20))



# Возможные бизнесс фичи (ПЕРЕСМОТРЕТЬ!)
# --- График 1: Доля дефолтов по возрастным бакетам ---
df['AGE_YEARS'] = -df['DAYS_BIRTH'] / 365
ax = df.groupby(pd.cut(df['AGE_YEARS'], bins=[20,30,40,50,60,70]), observed=True)['TARGET'].mean().plot(
    kind='bar', 
    figsize=(8, 5),
    color='steelblue',
    edgecolor='black'
)
ax.set_title('Доля дефолтов в зависимости от возраста клиента', fontsize=14)
ax.set_xlabel('Возрастная группа (лет)', fontsize=12)
ax.set_ylabel('Доля дефолтов (TARGET)', fontsize=12)
ax.set_xticklabels(['20-30', '30-40', '40-50', '50-60', '60-70'], rotation=0)
plt.tight_layout()
plt.savefig('reports/age_default_rate.png', dpi=150) # Сохранение
plt.close() # Закрываем фигуру


# --- График 2: Доля дефолтов по образованию ---
ax = df.groupby('NAME_EDUCATION_TYPE')['TARGET'].mean().sort_values().plot(
    kind='barh', 
    figsize=(10, 6),
    color='coral',
    edgecolor='black'
)
ax.set_title('Доля дефолтов в зависимости от уровня образования', fontsize=14)
ax.set_xlabel('Доля дефолтов (TARGET)', fontsize=12)
ax.set_ylabel('Уровень образования', fontsize=12)
plt.tight_layout()
plt.savefig('reports/education_default_rate.png', dpi=150)
plt.close()


# --- График 3: Распределение дохода ---
# Переводим доход в тысячи для красоты
df['INCOME_K'] = df['AMT_INCOME_TOTAL'] / 1000

ax = df[df.TARGET==0]['INCOME_K'].plot(kind='kde', label='Нет дефолта', color='steelblue', linewidth=2)
df[df.TARGET==1]['INCOME_K'].plot(kind='kde', label='Дефолт', color='coral', linewidth=2)
ax.set_title('Распределение дохода клиентов', fontsize=14)
ax.set_xlabel('Годовой доход (тыс. у.е.)', fontsize=12)
ax.set_ylabel('Плотность распределения', fontsize=12)
ax.set_xscale('log') # Логарифмическая шкала
ax.set_xlim(20, 1000) # Ограничим "хвост" выбросов для наглядности
ax.legend(title='Статус')
plt.tight_layout()
plt.savefig('reports/income_distribution.png', dpi=150)
plt.close()

# --- График 4: проверка безработицы ---
#n_anomaly = (df['DAYS_EMPLOYED'] == 365243).sum()
#print(f"Доля строк с заглушкой DAYS_EMPLOYED (365243): {n_anomaly / len(df):.2%}")

#print("Доля дефолтов среди безработных (заглушка):", df[df['DAYS_EMPLOYED'] == 365243]['TARGET'].mean())
#print("Доля дефолтов среди работающих:", df[df['DAYS_EMPLOYED'] != 365243]['TARGET'].mean())

# --- График 5: проверка корреляции долг/доход ---
cols_to_check = ['AMT_INCOME_TOTAL', 'AMT_CREDIT', 'AMT_ANNUITY', 'AMT_GOODS_PRICE', 'TARGET']
sns.heatmap(df[cols_to_check].corr(), annot=True, cmap='coolwarm', fmt=".2f")
plt.title("Корреляция ключевых сумм")
plt.savefig('reports/corr_amounts.png', dpi=150, bbox_inches='tight')