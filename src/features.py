import pandas as pd 
import numpy as np



""""Считаем кредитную репутацию клиента на основе прошлых и ативных кредитов"""
def aggregate_bureau(bureau_path='data/bureau.csv'):
    print("Агрегация bureau.csv...")
    bureau = pd.read_csv(bureau_path)

    # 1. Базовые агрегаты по всем кредитам
    agg_funcs = {
        'SK_ID_BUREAU': ['count'], 
        'DAYS_CREDIT': ['min', 'max', 'mean'], # как давно брал кредиты
        'CREDIT_DAY_OVERDUE': ['max', 'mean'], # просрочки
        'AMT_CREDIT_SUM': ['sum', 'mean', 'max'], # суммы кредитов
        'AMT_CREDIT_SUM_DEBT': ['sum', 'mean'], # текущий долг
        'AMT_CREDIT_SUM_OVERDUE': ['sum', 'mean'], # просроченный долг
        'CNT_CREDIT_PROLONG': ['sum'] # сколько раз продлевали
    }
    
    bureau_agg = bureau.groupby('SK_ID_CURR').agg(agg_funcs)
    bureau_agg.columns = ['BUREAU_' + '_'.join(col).upper() for col in bureau_agg.columns]
    
    # 2. Агрегаты только по АКТИВНЫМ кредитам (кредиты, которые еще платятся)
    active = bureau[bureau['CREDIT_ACTIVE'] == 'Active']
    active_agg = active.groupby('SK_ID_CURR').agg({
        'SK_ID_BUREAU': ['count'],
        'AMT_CREDIT_SUM_DEBT': ['sum', 'mean'],
        'AMT_CREDIT_SUM_OVERDUE': ['sum']
    })
    active_agg.columns = ['BUREAU_ACTIVE_' + '_'.join(col).upper() for col in active_agg.columns]
    
    # Объединяем
    bureau_features = bureau_agg.join(active_agg, how='left').reset_index()
    
    print(f"Создано {bureau_features.shape[1] - 1} фичей из bureau.csv")
    return bureau_features




""""Кредитное поведение клиента на основе кредитных заявок"""
def aggregate_previous_application(prev_path='data/previous_application.csv'):
    print("Агрегация previous_application.csv...")
    prev = pd.read_csv(prev_path)
    
    # 1. Базовые агрегаты
    agg_funcs = {
        'SK_ID_PREV': ['count'],
        'AMT_APPLICATION': ['mean', 'max', 'sum'], # сколько просил
        'AMT_CREDIT': ['mean', 'max', 'sum'],      # сколько дали
        'DAYS_DECISION': ['min', 'max', 'mean'],   # как давно принимал решение
        'CNT_PAYMENT': ['mean']                    # средний срок кредита
    }
    prev_agg = prev.groupby('SK_ID_CURR').agg(agg_funcs)
    prev_agg.columns = ['PREV_' + '_'.join(col).upper() for col in prev_agg.columns]
    
    # 2. Считаем статусы (одобрено, отказано, отменено)
    prev['IS_APPROVED'] = (prev['NAME_CONTRACT_STATUS'] == 'Approved').astype(int)
    prev['IS_REFUSED'] = (prev['NAME_CONTRACT_STATUS'] == 'Refused').astype(int)
    
    status_agg = prev.groupby('SK_ID_CURR').agg({
        'IS_APPROVED': ['sum', 'mean'], # сколько одобрений и их доля
        'IS_REFUSED': ['sum', 'mean']   # сколько отказов и их доля
    })
    status_agg.columns = ['PREV_STATUS_' + '_'.join(col).upper() for col in status_agg.columns]
    
    # 3. Разница между запрошенной и выданной суммой (насколько банк урезал кредит)
    prev['CREDIT_APP_DIFF'] = prev['AMT_APPLICATION'] - prev['AMT_CREDIT']
    diff_agg = prev.groupby('SK_ID_CURR')['CREDIT_APP_DIFF'].agg(['mean', 'max'])
    diff_agg.columns = ['PREV_DIFF_' + col.upper() for col in diff_agg.columns]
    
    # Объединяем
    prev_features = prev_agg.join(status_agg).join(diff_agg).reset_index()
    
    print(f"Создано {prev_features.shape[1] - 1} фичей из previous_application.csv")
    return prev_features




""""Объединяем фичи с 'application_train' по столбцу 'SK_ID_CURR'"""
if __name__ == "__main__":

    app = pd.read_csv('data/application_train.csv')
    
    # Создаем фичи
    bureau_feats = aggregate_bureau()
    prev_feats = aggregate_previous_application()
    
    # Объединяем по клиенту (left join, чтобы не потерять клиентов без истории)
    print("Объединение таблиц...")
    df_final = app.merge(bureau_feats, on='SK_ID_CURR', how='left')
    df_final = df_final.merge(prev_feats, on='SK_ID_CURR', how='left')
    
    print(f"Итоговый размер датасета: {df_final.shape}")
    
    # Сохраняем результат
    df_final.to_csv('data/application_train_features.csv', index=False)
    print("Данные сохранены в data/application_train_features.csv")