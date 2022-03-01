from natasha import NewsEmbedding, NewsMorphTagger, Segmenter, Doc
from dostoevsky.tokenization import RegexTokenizer
from dostoevsky.models import FastTextSocialNetworkModel


story_to_merge = ['party_rk', 'date_time', 'category', 'neutral', 'positive', 'negative',
                  'event_dislike', 'event_favorite', 'event_like']

columns_to_train_text = ['Супермаркеты', 'Рестораны',
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
                         'cur_cat_Транспорт', 'cur_cat_Фаст Фуд', 'cur_cat_Финансовые услуги',
                         'category', 'neutral', 'positive', 'negative',
                         'event_dislike', 'event_favorite', 'event_like']


def sentiment(x, model, Doc, segmenter, morph_tagger):
    """
    Sentiment analysis for single text string
    :param x: inpu string
    :param model: dostoevsky model
    :param Doc: natasha's instance for input
    :param segmenter: natasha's instance for segmentation
    :param morph_tagger: natasha's instance for tagging
    :return: list of predictions with probabilities
    """
    text = x.replace('\xa0', '')[5:-2]
    doc = Doc(text)
    doc.segment(segmenter)
    doc.tag_morph(morph_tagger)
    em = ''
    for d in doc.tokens:
        if d.pos != 'PUNCT':
            em = em + ' ' + d.text
    em = em[1:]
    preds = model.predict([em], k=5)
    return preds


def make_predictions(df):
    """
    Sentiment analysis for all contained texts in DataFrame.
    :param df: input DataFrame
    :return: DataFrame with three features of model output
    """
    segmenter = Segmenter()
    emb = NewsEmbedding()
    morph_tagger = NewsMorphTagger(emb)
    tokenizer = RegexTokenizer()
    model = FastTextSocialNetworkModel(tokenizer=tokenizer)
    neu = []
    pos = []
    neg = []
    for item in df.story_text.values:
        pr = sentiment(item, model, Doc, segmenter, morph_tagger)
        out_ar = [pr[0]['neutral'], pr[0]['positive'], pr[0]['negative']]
        neu.append(out_ar[0])
        pos.append(out_ar[1])
        neg.append(out_ar[2])
        if len(neu) % 500 == 0 and len(neu) != 0:
            print(f'Done {len(neu)} from {df.shape[0]}')
    df['neutral'] = neu
    df['positive'] = pos
    df['negative'] = neg
    return df