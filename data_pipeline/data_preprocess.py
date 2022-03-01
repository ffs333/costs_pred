import pandas as pd

from data_pipeline.data_utils import sep_by_dates, gender, mart, \
    columns_to_train, for_one, story_date
from utils.text_utils import make_predictions, story_to_merge, columns_to_train_text
from utils.utils import train_eval_split


def prepare_data(data_config):
    """
    Data preparing function
    :param data_config: data config
    :return:
    """
    train_trans = pd.read_csv(data_config.train_tr)
    test_trans = pd.read_csv(data_config.test_tr)
    socdem = pd.read_csv(data_config.socdem)
    if data_config.use_texts:
        story_logs = pd.read_csv(data_config.story_logs)
        story_text = pd.read_csv(data_config.story_texts)
    print('All data downloaded.\n'
          'Start preprocessing pipeline.')
    full_df = pd.concat([train_trans, test_trans]).reset_index(drop=True)

    months = list(set([x[:7] for x in list(set(full_df.transaction_dttm.values))]))
    months.sort()

    full_df = sep_by_dates(full_df, months)

    mns = 8
    final = []
    ids = list(set(full_df.party_rk.values))
    for i in range(mns, len(months)):
        print(f'{months[i - mns]} - {months[i - 1]}  eval {months[i]}')
        sub_df = full_df[(full_df.date >= f'{months[i - mns]}') & (full_df.date <= f'{months[i]}')]

        for id_ in ids:
            final.extend(for_one(sub_df[sub_df['party_rk'] == id_]))

    full_df = pd.DataFrame(final)

    full_df = full_df.merge(socdem, how='inner', on='party_rk')

    full_df['gender_cd'] = full_df['gender_cd'].apply(gender)
    full_df['marital_status_desc'] = full_df['marital_status_desc'].apply(mart)
    full_df = pd.get_dummies(full_df, columns=['cur_CAT'], prefix='cur_cat')

    if not data_config.use_texts:
        print(f'Data processing finished.')
        return train_eval_split(full_df, months, columns_to_train, data_config.save_prepared)

    print('Text processing starts.')
    story = story_logs.merge(story_text, how='inner', on='story_id')
    story = story[story.event.isin(['like', 'dislike', 'favorite'])]
    story['date_time'] = story['date_time'].apply(story_date)

    story = make_predictions(story)
    story = pd.get_dummies(story, columns=['event'], prefix='event')
    story = story.drop_duplicates(keep='last')

    full_df = full_df.merge(story[story_to_merge], how='inner', on='party_rk')
    full_df = full_df.drop_duplicates(keep='last')
    print(f'Data processing finished.')
    return train_eval_split(full_df, months, columns_to_train_text, data_config.save_prepared)

