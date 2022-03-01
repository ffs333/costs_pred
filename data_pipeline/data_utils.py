import pandas as pd
import numpy as np


columns_to_train = ['Супермаркеты', 'Рестораны',
                    'Сервисные услуги', 'Одежда/Обувь', 'Транспорт', 'Связь/Телеком',
                    'Финансовые услуги', 'Разные товары', 'Автоуслуги', 'Дом/Ремонт',
                    'Красота', 'Фаст Фуд', 'Топливо', 'Наличные', 'Аптеки', 'spends_mon0',
                    'spends_mon1', 'spends_mon2', 'spends_mon3', 'spends_mon4',
                    'spends_mon5', 'spends_mon6', 'spends_mon7', 'label', 'gender_cd',
                    'age', 'marital_status_desc', 'children_cnt', 'region_flg',
                    'cur_cat_Автоуслуги', 'cur_cat_Аптеки', 'cur_cat_Дом/Ремонт',
                    'cur_cat_Красота', 'cur_cat_Наличные', 'cur_cat_Одежда/Обувь',
                    'cur_cat_Разные товары', 'cur_cat_Рестораны', 'cur_cat_Связь/Телеком',
                    'cur_cat_Сервисные услуги', 'cur_cat_Супермаркеты', 'cur_cat_Топливо',
                    'cur_cat_Транспорт', 'cur_cat_Фаст Фуд', 'cur_cat_Финансовые услуги']


def sep_by_dates(df, months):
    """
    Split data by dates and categories
    :param df: input dataframe
    :param months: months in data
    :return: separated dataframe
    """
    out = []
    user_ids = list(set(df.party_rk.values))
    lenn = len(user_ids)

    for idx in range(len(user_ids)):
        user_df = df[df.party_rk == user_ids[idx]]
        for mon in months:
            mon_df = user_df.iloc[list(np.where(user_df.transaction_dttm.str.contains(mon))[0])]

            cats = ['Супермаркеты', 'Рестораны', 'Сервисные услуги', 'Одежда/Обувь', 'Транспорт', 'Связь/Телеком',
                    'Финансовые услуги',
                    'Разные товары', 'Автоуслуги', 'Дом/Ремонт', 'Красота', 'Фаст Фуд', 'Топливо', 'Наличные', 'Аптеки']
            for i in range(len(cats)):
                sub = mon_df[mon_df.category == cats[i]]
                count = sub.shape[0]
                summa = sum(sub.transaction_amt_rur.values)
                out.append({
                    'party_rk': user_ids[idx],
                    'date': mon,
                    'category': cats[i],
                    'count_tr': count,
                    'summ': summa
                })
        if idx % 300 == 0 and idx != 0:
            print(f'Done {idx} user ids from {lenn}')

    return pd.DataFrame(out)


def for_one(subb):
    """
    Aggregate all categories to one row by date period
    :param subb: sub data frame
    :return: list of all categories spends
    """
    cats = ['Супермаркеты', 'Рестораны', 'Сервисные услуги', 'Одежда/Обувь', 'Транспорт', 'Связь/Телеком',
            'Финансовые услуги', 'Разные товары', 'Автоуслуги', 'Дом/Ремонт', 'Красота', 'Фаст Фуд', 'Топливо',
            'Наличные', 'Аптеки']
    base_d = {
        'party_rk': subb.party_rk.values[0],
        'start': min(subb.date.values),
        'end': max(subb.date.values)
    }
    output = []
    for item in cats:
        cc = sum(subb[(subb.category == item) & (subb.date < base_d['end'])].count_tr.values)
        base_d[f'{item}'] = cc
    for item in cats:
        sums = subb[subb.category == item].summ.values
        out_d = base_d.copy()
        out_d['cur_CAT'] = item
        for i in range(len(sums)):
            if i == len(sums) - 1:
                out_d['label'] = sums[i]
            else:
                out_d[f'spends_mon{i}'] = sums[i]
        output.append(out_d)
    return output


def gender(x):
    return 1 if x == 'M' else 0


def mart(x):
    if x == 'Вдовец, вдова':
        return 0
    elif x == 'Гражданский брак':
        return 1
    elif x == 'Разведен (а)':
        return 2
    elif x == 'Холост/не замужем':
        return 3
    elif x == 'Женат/замужем':
        return 4
    else:

        return 5


def story_date(x):
    return x.split('-')[0] + '-' + x.split('-')[1]