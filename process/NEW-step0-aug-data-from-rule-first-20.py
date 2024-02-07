# step1: get the 100 next_item_predictions for each item in train_data using rule-baseline method

import warnings
warnings.simplefilter('ignore')

import gc
import re
from collections import defaultdict, Counter

import pickle
import random
import numpy as np
import pandas as pd
pd.set_option('display.max_columns', None)
from tqdm.auto import tqdm

import torch

df_prod = pd.read_csv('products_train.csv')
df_prod

df_sess = pd.read_csv('sessions_train.csv')
df_sess


df_test = pd.read_csv('sessions_test_task1_phase2.csv')
df_test


def str2list(x):
    x = x.replace('[', '').replace(']', '').replace("'", '').replace('\n', ' ').replace('\r', ' ').replace('\t', ' ').replace(',', ' ')
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


# ------ 由原数据生成新数据：随机切分为长度在1~20间的session段：moco_df_sess ------ #


# 切分训练集session的prev_items，取最后一个商品ID作为last_item，切分长度为5的session

moco_df_sess = pd.DataFrame()
for i, row in tqdm(df_sess.iterrows(), total=len(df_sess)):
    prev_items = str2list(row['prev_items'])
    prev_items_length = len(prev_items)
    if prev_items_length <= 20:
        moco_df_sess = pd.concat([moco_df_sess, row.to_frame().T])
    else:
        # cut one session into several sessions of length not larger than 5
        for j in range(0, prev_items_length-20):
            new_row = row.copy()
            row_len = random.randint(1, 20)
            new_row['prev_items'] = str(prev_items[j:j+row_len])
            new_row['next_item'] = prev_items[j+row_len]
            moco_df_sess = pd.concat([moco_df_sess, new_row.to_frame().T])
        # add the last session
        new_row = row.copy()
        row_len = random.randint(1, 20)
        new_row['prev_items'] = str(prev_items[-row_len:])
        new_row['next_item'] = row['next_item']
        moco_df_sess = pd.concat([moco_df_sess, new_row.to_frame().T])
    if i % 10000 == 0 and i != 0:
        # save moco_df_sess into csv file using id i
        moco_df_sess = moco_df_sess.reset_index(drop=True)
        moco_df_sess.to_csv('moco_df_sess_{}.csv'.format((int)(i/10000)), index=False)
        moco_df_sess = pd.DataFrame()


# save the last moco_df_sess into csv file
moco_df_sess = moco_df_sess.reset_index(drop=True)
moco_df_sess.to_csv('moco_df_sess_{}.csv'.format((int)(len(df_sess)/10000)+1), index=False)



# load all moco_df_sess and concat them
new_parts = len(df_sess) // 10000 + 1
moco_df_sess = pd.DataFrame()
for i in tqdm(range(1, new_parts+1)):
    moco_df_sess = pd.concat([moco_df_sess, pd.read_csv('moco_df_sess_{}.csv'.format(i))])


moco_df_sess = moco_df_sess.reset_index(drop=True)
moco_df_sess


# save moco_df_sess
moco_df_sess.to_csv('moco_df_sess.csv', index=False)
# # load moco_df_sess
# moco_df_sess = pd.read_csv('moco_df_sess.csv')






# ------ 由moco_df_sess生成新数据：预测下一个商品ID：moco_df_sess ------ #

# 找出训练集session的最后一个商品ID并根据next_item_map查表预测

moco_df_sess['last_item'] = moco_df_sess['prev_items'].apply(lambda x: str2list(x)[-1])

for locale in tqdm(moco_df_sess['locale'].unique()):
    moco_df_sess_locale = moco_df_sess[moco_df_sess['locale'] == locale]
    moco_df_sess_locale['next_item_prediction'] = moco_df_sess_locale['last_item'].apply(lambda x: next_item_map_dict[locale][x] if x in next_item_map_dict[locale] else [])
    moco_df_sess.loc[moco_df_sess['locale'] == locale, 'next_item_prediction'] = moco_df_sess_locale['next_item_prediction']

moco_df_sess

# 若预测结果为空，则取top200中的商品前100个
# 若预测结果不足100个，则将top200中的商品按顺序填充至100个（除去重复和已交互商品）

preds = []

for _, row in tqdm(moco_df_sess.iterrows(), total=len(moco_df_sess)):
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


moco_df_sess['next_item_prediction'] = preds
moco_df_sess

moco_df_sess['next_item_prediction'].apply(len).describe()

moco_df_sess.to_pickle('sep_train_rule_recall_100_all.pkl')
# moco_df_sess = pd.read_pickle('sep_train_rule_recall_100.pkl')

moco_df_sess[['locale', 'next_item_prediction']].to_pickle('sep_train_rule_recall_100.pkl')
# moco_df_sess = pd.read_pickle('sep_train_rule_recall_100.pkl')




# ----- Get the corresponding train_data from moco_df_sess ----- #


# convert prev_items from string to list
moco_df_sess['prev_items_list'] = moco_df_sess['prev_items'].apply(lambda x: str2list(x))


# load the feature tensor of all items
nontext_features = torch.load('nontext_features.pt')


# make a map from item to its feature tensor
# to fasten search the corresponding feature tensor
item2feature = {}
for i, row in tqdm(df_prod.iterrows(), total=len(df_prod)):
    # get numpy array of feature tensor feature_tensor[i]
    item2feature[str(row['id']) + ' ' + str(row['locale'])] = nontext_features[i].cpu().numpy()


def get_feature(item, locale):
    # find feature tensor for feature_tensor
    # check if item is in item2feature
    if (item + ' ' + locale) not in item2feature:
        print('item {} not in item2feature'.format(item + ' ' + locale))
        return []
    feature = item2feature[item + ' ' + locale]
    return feature


def get_feature_list(prev_items_list, next_item, locale):
    feature_list = []
    # get the feature tensor for prev_items_list
    for item in prev_items_list:
        feature_list.append(get_feature(item, locale))
    # get the feature tensor for next_item
    feature_list.append(get_feature(next_item, locale))
    return feature_list


def get_feature_singlelist(items_list, locale):
    feature_list = []
    # get the feature tensor for prev_items_list
    for item in items_list:
        feature_list.append(get_feature(item, locale))
    return feature_list



# get the 1st part of train_data: prev_items (refer to 'query') : get feature_list of row['prev_items_list'] & row['locale']
moco_df_sess['prev_items_feature_list'] = None
for _, row in tqdm(moco_df_sess.iterrows(), total=len(moco_df_sess)):
    row['prev_items_feature_list'] = get_feature_singlelist(row['prev_items_list'], row['locale'])


# get the 2nd part of train_data: next_item (refer to 'label') : get feature_list of row['next_item']
moco_df_sess['next_item_feature_list'] = None 
for _, row in tqdm(moco_df_sess.iterrows(), total=len(moco_df_sess)):
    row['next_item_feature_list'] = get_feature(row['next_item'], row['locale'])


# get the 3rd part of train_data: next_item_prediction (refer to 'keys') : get feature_list of row['next_item_prediction']
moco_df_sess['next_item_prediction_feature_list'] = None
for _, row in tqdm(moco_df_sess.iterrows(), total=len(moco_df_sess)):
    row['next_item_prediction_feature_list'] = get_feature_singlelist(row['next_item_prediction'], row['locale'])


# generate the 4th part of train_data: samples (refer to 'samples') : get feature_list of row['samples']
# the samples of each row got 100 samples: the 1st one is the next_item, the rest are first items in next_item_prediction except the next_item
moco_df_sess['samples_feature_list'] = moco_df_sess['next_item_feature_list']

for i, row in tqdm(moco_df_sess.iterrows(), total=len(moco_df_sess)):
    samples_feature_list = row['samples_feature_list']
    for item in row['next_item_prediction']:
        if item != row['next_item']:
            # samples_feature_list is np.ndarray
            samples_feature_list = np.vstack((samples_feature_list, get_feature(item, row['locale'])))
            # samples_feature_list.append(get_feature(item, row['locale']))
        if len(samples_feature_list) >= 100:
            break
    if len(samples_feature_list) < 100:
        print(f'row {i} has less than 100 samples!')
    row['samples_feature_list'] = samples_feature_list


# save moco_df_sess[['prev_items_feature_list', 'next_item_feature_list', 'next_item_prediction_feature_list', 'samples_feature_list']] together to file by pickle
with open('moco_feature_list.pkl', 'wb') as f:
    pickle.dump(moco_df_sess[['prev_items_feature_list', 'next_item_feature_list', 'next_item_prediction_feature_list', 'samples_feature_list']], f)
# # load moco_feature_list
# with open('moco_feature_list.pkl', 'rb') as f:
#     moco_feature_list = pickle.load(f)

