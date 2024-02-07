import warnings
warnings.simplefilter('ignore')

import gc
import re
from collections import defaultdict, Counter

import random
from tqdm import *
import pickle
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

pd.set_option('display.max_columns', None)
from tqdm.auto import tqdm

import torch
import torch.nn.functional as F

df_prod = pd.read_csv('products_train.csv')
df_prod

df_sess = pd.read_csv('sessions_train.csv')
df_sess


df_test = pd.read_csv('sessions_test_task1.csv')
df_test

# # get first line of df_sess
# df_sess.iloc[0]

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

# 找出测试集session的最后一个商品ID并根据next_item_map查表预测


# construct the validation set by choosing 10000 random samples from df_sess
df_validate = df_sess.sample(10000, random_state=2021)
# keep the original index in df_sess in df_validate as a column named 'id'
df_validate['id'] = df_validate.index
# reset the index of df_validate
df_validate.reset_index(drop=True, inplace=True)

df_validate['last_item'] = df_validate['prev_items'].apply(lambda x: str2list(x)[-1])
df_validate['next_item_prediction'] = df_validate['last_item'].map(next_item_map)
df_validate

# 若预测结果为空，则取top200中的商品前100个
# 若预测结果不足100个，则将top200中的商品按顺序填充至100个（除去重复和已交互商品）

preds = []

for _, row in tqdm(df_validate.iterrows(), total=len(df_validate)):
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


df_validate['next_item_prediction'] = preds
df_validate


df_validate['next_item_prediction'].apply(len).describe()


# filter the item with value larger than 3570000
validate_list = [i for i in df_validate['id'].tolist() if i < 3570000]

# get the new df_validate using id in validate_list
df_validate = df_validate[df_validate['id'].isin(validate_list)]

df_validate['prev_items_list'] = df_validate['prev_items'].apply(lambda x: str2list(x))


# # save df_validate[['locale', 'next_item_prediction']] into pickle
# df_validate[['locale', 'next_item_prediction']].to_pickle('rule_recall_100_validate_preds.pkl')
# df_validate['id'].to_pickle('rule_recall_100_validate_data.pkl')
df_validate.to_pickle('rule_recall_100_validate.pkl')


# calculate the MRR score of df_validate
mrr = 0
# curr_mrr = []
for _, row in tqdm(df_validate.iterrows(), total=len(df_validate)):
    if row['next_item'] in row['next_item_prediction']:
        mrr_score = 1 / (row['next_item_prediction'].index(row['next_item']) + 1)
        mrr += mrr_score
        # print(f'row {row["id"]} is correct, MRR score is: {mrr_score}')
        # curr_mrr.append(mrr_score)

mrr /= len(df_validate)
print('MRR score of df_validate is: ', mrr)
# # plot the distribution of mrr score
# plt.hist(curr_mrr, bins=200)
# plt.savefig('mrr_score_distribution.png')



# write a function to calculate the MRR score
def calculate_mrr(df):
    mrr = 0
    for _, row in tqdm(df.iterrows(), total=len(df)):
        if row['next_item'] in row['next_item_prediction']:
            mrr_score = 1 / (row['next_item_prediction'].index(row['next_item']) + 1)
            mrr += mrr_score
    mrr /= len(df)
    return mrr



# # load the train_data
# train_data = torch.load('train_data.pt')
# # load the train_mask
# train_mask = torch.load('train_mask.pt')




# # shape of validate_list: (10000, 20, 28)
# validate_data = train_data[validate_list]
# # shape of validate_mask: (10000, 20)
# validate_mask = train_mask[validate_list]


# # count the mask_length of each sample in validate_mask
# mask_length = validate_mask.sum(dim=1)


# prev_items = []
# next_item = []
# # mask the validate_data with validate_mask to get the prev_items and next_item
# for i in range(len(validate_data)):
#     # shape of prev_items: (10000, mask_length - 1)
#     prev_items.append(validate_data[i][validate_mask[i].bool()][:-1, :])
#     # shape of next_item: (10000, 1)
#     next_item.append(validate_data[i][validate_mask[i].bool()][-1, :])
    

#     # get the id of the next_item
#     next_item_id = next_item[i][-1]









# # generate validation set
# def str2list(x):
#     x = x.replace('[', '').replace(']', '').replace("'", '').replace('\n', ' ').replace('\r', ' ')
#     l = [i for i in x.split() if i]
#     return l




# df_validate['prev_items_list'] = df_validate['prev_items'].apply(lambda x: str2list(x))
# df_validate['prev_items_list'].apply(lambda x: len(x)).max()

# nontext_features = torch.load('nontext_features.pt')



# # make a map from item to its feature tensor
# # to fasten search the corresponding feature tensor
# item2feature = {}
# for i, row in tqdm(df_prod.iterrows(), total=len(df_prod)):
#     # get numpy array of feature tensor feature_tensor[i]
#     item2feature[str(row['id']) + ' ' + str(row['locale'])] = nontext_features[i].cpu().numpy()



# # get features by id + locale
# in_prods = 0
# out_prods = 0
# def get_feature(item, locale):
#     # find feature tensor for feature_tensor
#     feature = None
#     if (item + ' ' + locale) in item2feature:
#         feature = item2feature[item + ' ' + locale]
#         global in_prods
#         in_prods += 1
#     else:
#         global out_prods
#         out_prods += 1
#     return feature



# def get_feature_list(prev_items_list, next_item, locale):
#     feature_list = []
#     # get the feature tensor for prev_items_list
#     for item in prev_items_list:
#         if get_feature(item, locale) is not None:
#             feature_list.append(get_feature(item, locale))
#     # get the feature tensor for next_item
#     if get_feature(next_item, locale) is not None:
#         feature_list.append(get_feature(next_item, locale))
#     return feature_list



# # get feature list for each row of df_validate and corresponding next_item_prediction
# validate_feature_list = []
# next_item_prediction = df_validate['next_item_prediction']
# for i, row in tqdm(df_validate.iterrows(), total=len(df_validate)):
#     test_feature_list = []
#     for item in next_item_prediction[i]:
#         my_feature = get_feature_list(row['prev_items_list'], item, row['locale'])
#         test_feature_list.append(my_feature)
#     validate_feature_list.append(test_feature_list)


# # generate valid_data


# # convert validate_feature_list to pandas DataFrame
# validate_feature_list = pd.DataFrame(validate_feature_list)


# # cut the validate_feature_list to maxlen in the 3rd dimension
# maxlen = 20
# final_validate_data = validate_feature_list.apply(lambda x: x.apply(lambda y: y[: (min(len(y), maxlen)) ]))


# # convert final_validate_data to torch.Tensor
# final_validate_data = final_validate_data.apply(lambda x: x.apply(lambda y: np.array(y)))
# final_validate_data = final_validate_data.apply(lambda x: x.apply(lambda y: torch.from_numpy(y)))
# # pad each item of final_validate_data using torch.nn.functional.pad on the 3th dimension with value 0
# final_validate_data = final_validate_data.apply(lambda x: x.apply(lambda y: F.pad(y, (0, 0, 0, maxlen - y.shape[0]), 'constant', 0)))


# # convert the final_validate_data to torch.tensor
# list1 = []
# for i in tqdm(range(len(final_validate_data))):
#     list2 = []
#     for j in range(len(final_validate_data.iloc[i])):
#         list2.append(final_validate_data.iloc[i][j])
#     list2 = torch.stack(list2)
#     list1.append(list2)

# final_validate_data = torch.stack(list1)


# # # save the final_validate_data
# # torch.save(final_validate_data, 'final_validate_data.pt')

# # # load the final_validate_data
# # final_validate_data = torch.load('final_validate_data.pt')


