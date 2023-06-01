import warnings
warnings.simplefilter('ignore')

import gc
import re
from collections import defaultdict, Counter

import numpy as np
import pandas as pd
pd.set_option('display.max_columns', None)
from tqdm.auto import tqdm


# df_prod = pd.read_csv('products_train.csv')
# df_prod

df_sess = pd.read_csv('sessions_train.csv')
df_sess


df_test = pd.read_csv('sessions_test_task1_phase2.csv')
df_test

# # get first line of df_sess
# df_sess.iloc[0]

def str2list(x):
    x = x.replace('[', '').replace(']', '').replace("'", '').replace('\n', ' ').replace('\r', ' ')
    l = [i for i in x.split() if i]
    return l


# get the most frequent 200 next_item for each locale
next_item_map_dict = defaultdict(list)
top200_dict = defaultdict(list)

for locale in df_sess['locale'].unique():
    # step1: calculate the next_item_dict for each locale
    next_item_dict_locale = defaultdict(list)
    # 1st part: calculate on df_sess
    df_sess_locale = df_sess[df_sess['locale'] == locale]
    for _, row in tqdm(df_sess_locale.iterrows(), total=len(df_sess_locale)):
        prev_items = str2list(row['prev_items'])
        next_item = row['next_item']
        prev_items_length = len(prev_items)
        if prev_items_length <= 1:
            next_item_dict_locale[prev_items[0]].append(next_item)
        else:
            for i, item in enumerate(prev_items[:-1]):
                next_item_dict_locale[item].append(prev_items[i+1])
            next_item_dict_locale[prev_items[-1]].append(next_item)
    # 2nd part: calculate on df_test
    df_test_locale = df_test[df_test['locale'] == locale]
    for _, row in tqdm(df_test_locale.iterrows(), total=len(df_test_locale)):
        prev_items = str2list(row['prev_items'])
        prev_items_length = len(prev_items)
        if prev_items_length <= 1:
            continue
        else:
            for i, item in enumerate(prev_items[:-1]):
                next_item_dict_locale[item].append(prev_items[i+1])
    # step2: calculate the next_item_map for each locale
    next_item_map_locale = {}
    for item in tqdm(next_item_dict_locale):
        counter = Counter(next_item_dict_locale[item])
        next_item_map_locale[item] = [i[0] for i in counter.most_common(100)]
    next_item_map_dict[locale] = next_item_map_locale
    # count the first 200 most frequent next_item for each item
    k = []
    v = []
    for item in next_item_dict_locale:
        k.append(item)
        v.append(next_item_dict_locale[item])
    df_next_locale = pd.DataFrame({'item': k, 'next_item': v})
    df_next_locale = df_next_locale.explode('next_item').reset_index(drop=True)
    top200_dict[locale] = df_next_locale['next_item'].value_counts().index.tolist()[:200]


# 找出测试集session的最后一个商品ID并根据next_item_map查表预测
df_test['last_item'] = df_test['prev_items'].apply(lambda x: str2list(x)[-1])

# get the df_test['next_item_prediction'] by looking up the next_item_map_dict for each locale using tqdm
for locale in tqdm(df_test['locale'].unique()):
    df_test_locale = df_test[df_test['locale'] == locale]
    df_test_locale['next_item_prediction'] = df_test_locale['last_item'].apply(lambda x: next_item_map_dict[locale][x] if x in next_item_map_dict[locale] else [])
    df_test.loc[df_test['locale'] == locale, 'next_item_prediction'] = df_test_locale['next_item_prediction']


# # find the number of empty list in df_test['next_item_prediction']
# df_test['next_item_prediction'].apply(lambda x: len(x)).value_counts().sort_index()


# 若预测结果为空，则取top200中的商品前100个
# 若预测结果不足100个，则将top200中的商品按顺序填充至100个（除去重复和已交互商品）

preds = []

for _, row in tqdm(df_test.iterrows(), total=len(df_test)):
    pred_orig = row['next_item_prediction']
    pred = pred_orig
    prev_items = str2list(row['prev_items'])
    if type(pred) == float:
        pred = top200_dict[row['locale']][:100]
    else:
        if len(pred_orig) < 100:
            for i in top200_dict[row['locale']]:
                if i not in pred_orig and i not in prev_items:
                    pred.append(i)
                if len(pred) >= 100:
                    break
        else:
            pred = pred[:100]
    preds.append(pred)


df_test['next_item_prediction'] = preds
df_test

df_test['next_item_prediction'].apply(len).describe()

# df_test[['locale', 'next_item_prediction']].to_parquet('rule_recall_test.parquet', engine='pyarrow')

# save df_test[['locale', 'next_item_prediction']] into pickle
df_test[['locale', 'next_item_prediction']].to_pickle('rule_recall_100_test.pkl')
# # load df_test[['locale', 'next_item_prediction']] from pickle
# df_test = pd.read_pickle('rule_recall_100_test.pkl')

