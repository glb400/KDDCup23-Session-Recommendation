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


df_test = pd.read_csv('sessions_test_task1.csv')
df_test


def str2list(x):
    x = x.replace('[', '').replace(']', '').replace("'", '').replace('\n', ' ').replace('\r', ' ')
    l = [i for i in x.split() if i]
    return l


# next_item_dict：存储用户session商品中所有的next关系
next_item_dict = defaultdict(list)


# 遍历存储所有的next关系
for _, row in tqdm(df_sess.iterrows(), total=len(df_sess)):
    prev_items = str2list(row['prev_items'])
    next_item = row['next_item']
    prev_items_length = len(prev_items)
    if prev_items_length <= 1:
        next_item_dict[prev_items[0]].append(next_item)
    else:
        for i, item in enumerate(prev_items[:-1]):
            next_item_dict[item].append(prev_items[i+1])
        next_item_dict[prev_items[-1]].append(next_item)

for _, row in tqdm(df_test.iterrows(), total=len(df_test)):
    prev_items = str2list(row['prev_items'])
    prev_items_length = len(prev_items)
    if prev_items_length <= 1:
        continue
    else:
        for i, item in enumerate(prev_items[:-1]):
            next_item_dict[item].append(prev_items[i+1])

# 统计各商品后next出现的各商品次数
# 将出现次数最多的100个商品作为预测结果存入next_item_map

next_item_map = {}

for item in tqdm(next_item_dict):
    counter = Counter(next_item_dict[item])
    next_item_map[item] = [i[0] for i in counter.most_common(100)]

# 将next_item_dict的统计结果按返回形式存入df_next

k = []
v = []

for item in next_item_dict:
    k.append(item)
    v.append(next_item_dict[item])


df_next = pd.DataFrame({'item': k, 'next_item': v})
df_next = df_next.explode('next_item').reset_index(drop=True)
df_next

# 找出最常作为next出现的200个商品ID

top200 = df_next['next_item'].value_counts().index.tolist()[:200]



# 切分训练集session的prev_items，取最后一个商品ID作为last_item，切分长度为5的session
new_df_sess = pd.DataFrame()
for i, row in tqdm(df_sess.iterrows(), total=len(df_sess)):
    prev_items = str2list(row['prev_items'])
    prev_items_length = len(prev_items)
    if prev_items_length <= 5:
        new_df_sess = pd.concat([new_df_sess, row.to_frame().T])
    else:
        # cut one session into several sessions of length not larger than 5
        for j in range(0, prev_items_length-5):
            new_row = row.copy()
            row_len = random.randint(1, 5)
            new_row['prev_items'] = str(prev_items[j:j+row_len])
            new_row['next_item'] = prev_items[j+row_len]
            new_df_sess = pd.concat([new_df_sess, new_row.to_frame().T])
        # add the last session
        new_row = row.copy()
        row_len = random.randint(1, 5)
        new_row['prev_items'] = str(prev_items[-row_len:])
        new_row['next_item'] = row['next_item']
        new_df_sess = pd.concat([new_df_sess, new_row.to_frame().T])
    if i % 10000 == 0 and i != 0:
        # save new_df_sess into csv file using id i
        new_df_sess = new_df_sess.reset_index(drop=True)
        new_df_sess.to_csv('new_df_sess_{}.csv'.format((int)(i/10000)), index=False)
        new_df_sess = pd.DataFrame()


# load all new_df_sess and concat them
new_parts = len(df_sess) // 20000 + 1
new_df_sess = pd.DataFrame()
for i in tqdm(range(1, new_parts+1)):
    new_df_sess = pd.concat([new_df_sess, pd.read_csv('new_df_sess_{}.csv'.format(i))])


new_df_sess = new_df_sess.reset_index(drop=True)
new_df_sess

# save new_df_sess
new_df_sess.to_csv('new_df_sess.csv', index=False)
# # load new_df_sess
# new_df_sess = pd.read_csv('new_df_sess.csv')



# 找出训练集session的最后一个商品ID并根据next_item_map查表预测

new_df_sess['last_item'] = new_df_sess['prev_items'].apply(lambda x: str2list(x)[-1])
new_df_sess['next_item_prediction'] = new_df_sess['last_item'].map(next_item_map)
new_df_sess

# 若预测结果为空，则取top200中的商品前100个
# 若预测结果不足100个，则将top200中的商品按顺序填充至100个（除去重复和已交互商品）

preds = []

for _, row in tqdm(new_df_sess.iterrows(), total=len(new_df_sess)):
    pred_orig = row['next_item_prediction']
    pred = pred_orig
    prev_items = str2list(row['prev_items'])
    if type(pred) == float:
        pred = top200[:100]
    else:
        if len(pred_orig) < 100:
            for i in top200:
                if i not in pred_orig and i not in prev_items:
                    pred.append(i)
                if len(pred) >= 100:
                    break
        else:
            pred = pred[:100]
    preds.append(pred)


new_df_sess['next_item_prediction'] = preds
new_df_sess

new_df_sess['next_item_prediction'].apply(len).describe()

# new_df_sess[['locale', 'next_item_prediction']].to_pickle('short_train_rule_recall_100_test.pkl')
new_df_sess.to_pickle('short_train_rule_recall_100_test.pkl')












# ------ Get the corresponding train_data from new_df_sess ------

new_df_sess['prev_items_list'] = new_df_sess['prev_items'].apply(lambda x: str2list(x))

nontext_features = torch.load('nontext_features.pt')

# make a map from item to its feature tensor
# to fasten search the corresponding feature tensor
item2feature = {}
for i, row in tqdm(df_prod.iterrows(), total=len(df_prod)):
    # get numpy array of feature tensor feature_tensor[i]
    item2feature[str(row['id']) + ' ' + str(row['locale'])] = nontext_features[i].cpu().numpy()


def get_feature(item, locale):
    # find feature tensor for feature_tensor
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


def get_feature_list2(prev_items_list, locale):
    feature_list = []
    # get the feature tensor for prev_items_list
    for item in prev_items_list:
        feature_list.append(get_feature(item, locale))
    return feature_list


# get the corresponding feature tensors for prev_items_list in df_sess according to their id & locale
# df_sess['feature_list'] = df_sess.apply(lambda x: get_feature_list(x['prev_items_list'], x['locale']), axis=1)


# using tqdm instead
# make a new column 'feature_list' in df_sess
new_df_sess['feature_list'] = None


# for _, row in tqdm(new_df_sess.iterrows(), total=len(new_df_sess)):
#     row['feature_list'] = get_feature_list(row['prev_items_list'], row['next_item'], row['locale'])



# !!!!!!
# get feature_list of row['prev_items_list'] & row['locale']
for _, row in tqdm(new_df_sess.iterrows(), total=len(new_df_sess)):
    row['feature_list'] = get_feature_list2(row['prev_items_list'], row['locale'])


# !!!!!!
# get feature_list of row['next_item_prediction']
new_df_sess['next_item_prediction_feature_list'] = new_df_sess['next_item_prediction'].apply(lambda x: [get_feature(i, row['locale']) for i in x])


with open('newsess_next_item_prediction_feature_list.pkl', 'wb') as f:
    pickle.dump(new_df_sess['next_item_prediction_feature_list'].values, f)





# save df_sess['feature_list'].values to file by pickle
with open('newsess_feature_list.pkl', 'wb') as f:
    pickle.dump(new_df_sess['feature_list'].values, f)











