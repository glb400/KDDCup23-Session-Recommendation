import warnings
warnings.simplefilter('ignore')

import gc
import re
import pickle
from collections import defaultdict, Counter

import numpy as np
import pandas as pd
pd.set_option('display.max_columns', None)
from tqdm.auto import tqdm

df_prod = pd.read_csv('products_train.csv')
df_prod

df_sess = pd.read_csv('sessions_train.csv')
df_sess

df_test = pd.read_csv('sessions_test_task1.csv')
df_test

# list all columns of df_prod
df_prod.columns
# Index(['id', 'locale', 'title', 'price', 'brand', 'color', 'size', 'model',
#        'material', 'author', 'desc'],
#       dtype='object')


import torch
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# feature1: id
# convert id to a non-repeating integer
id2int = {id: i for i, id in enumerate(df_prod['id'].unique())}
id2int
# save id2int to pickle file
with open('id2int.pkl', 'wb') as f:
    pickle.dump(id2int, f)
# # load id2int from pickle file
# with open('id2int.pkl', 'rb') as f:
#     id2int = pickle.load(f)

# convert id to integer according to id2int
id_num = torch.tensor(df_prod['id'].map(id2int).values).unsqueeze(1)
id_num

# save id_num to file
torch.save(id_num, 'id_num.pt')
# # load id_num from file
# id_num = torch.load('id_num.pt')


# ---

# feature2: locale

# generate one-hot encoding for locale
locale_onehot = pd.get_dummies(df_prod['locale'])
locale_onehot

# convert locale_onehot to pytorch tensor
locale_onehot = torch.tensor(locale_onehot.values)
locale_onehot

# save locale_onehot to file
torch.save(locale_onehot, 'locale_onehot.pt')
# # load locale_onehot from file
# locale_onehot = torch.load('locale_onehot.pt')


# ---


# feature4: price

# 1. discrete feature from price
# save price to file by pandas
df_prod['price'].to_csv('price.csv', index=False)

# load price from file by pandas
price = pd.read_csv('price.csv')
price

# plot the distribution of log(price+1) and save it to file
import matplotlib.pyplot as plt
plt.hist(df_prod['price'], bins=20)
plt.savefig('price_log_hist.png')

# count the number of 0 in price
price[price==0].count()

# count the number of nan in price
price.isna().sum()


# extract the id of rows in price that are outliers, i.e., price = 40000000.07 from price
price_outliers_id = price[price==40000000.07]
# drop the items with NaN values in price_outliers
price_outliers_id = price_outliers_id.dropna().index
price_outliers_id


# get the rest of the rows in price that are not outliers
price_inliers_id = price[price!=40000000.07]
# drop the items with NaN values in price_outliers
price_inliers_id = price_inliers_id.dropna().index
price_inliers_id


# use these indexes to get the corresponding rows in df_prod
price_outliers = price.iloc[price_outliers_id]
price_inliers = price.iloc[price_inliers_id]


# separate price_inliers into 20 bins with equal frequency
price_bins = pd.qcut(price_inliers['price'], 20)


# get the categories of Series price_bins
price_bins_list = price_bins.cat.categories.tolist()
# add one new bin to price_bins for price_outliers representing [3933800, 40000000.07]
price_bins_list.append(pd.Interval(left=3933800.0, right=40000000.07, closed='right'))


# convert price_bins_list to intervalIndex
price_bins_cat = pd.IntervalIndex(pd.Categorical(price_bins_list, ordered=True))


# use price_bins_cat to allocate a new feature to price
# this feature means item belongs to which bin according to the value of price['price']
price['price_bin'] = price['price'].apply(lambda x: price_bins_cat.get_loc(x))


# find a value is in which bin of price_bins_cat
price_bins_cat.get_loc(40000000.07)


# convert price['price_bin'] to one-hot encoding
price_onehot = pd.get_dummies(price['price_bin'])


price_onehot = torch.tensor(price_onehot.values, dtype=torch.float32)


# save price_all to file
torch.save(price_onehot, 'price_onehot.pt')
# # load price_all from file
# price_onehot = torch.load('price_onehot.pt')




# concat the tensors to a new tensor in dim=1
id_num = id_num.to(device) # 0
locale_onehot = locale_onehot.to(device) # 1-6
price_onehot = price_onehot.to(device) # 7-27
nontext_features = torch.cat((id_num, locale_onehot, price_onehot), dim=1)


# save feature_tensor to file
torch.save(nontext_features, 'nontext_features.pt')
# # load feature_tensor from file
# nontext_features = torch.load('nontext_features.pt')



# ---- NOW get feature tensor for each session ----

def str2list(x):
    x = x.replace('[', '').replace(']', '').replace("'", '').replace('\n', ' ').replace('\r', ' ')
    l = [i for i in x.split() if i]
    return l

df_sess['prev_items_list'] = df_sess['prev_items'].apply(lambda x: str2list(x))


df_sess['prev_items_list'].apply(lambda x: len(x)).max() # maxlen: 474




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


# get the corresponding feature tensors for prev_items_list in df_sess according to their id & locale
# df_sess['feature_list'] = df_sess.apply(lambda x: get_feature_list(x['prev_items_list'], x['locale']), axis=1)

# using tqdm instead
# make a new column 'feature_list' in df_sess
df_sess['feature_list'] = None

for _, row in tqdm(df_sess.iterrows(), total=len(df_sess)):
    row['feature_list'] = get_feature_list(row['prev_items_list'], row['next_item'], row['locale'])


# save df_sess['feature_list'].values to file by pickle
with open('feature_list.pkl', 'wb') as f:
    pickle.dump(df_sess['feature_list'].values, f)

# load df_sess['feature_list'].values from file by pickle
final_feature_list = None
with open('feature_list.pkl', 'rb') as f:
    final_feature_list = pickle.load(f)


# # # to sperately read into 20 parts
# part_len = len(df_sess) // 20
# for i in range(20):
#     df_sess_part = df_sess.iloc[i*part_len:(i+1)*part_len]
#     for _, row in tqdm(df_sess_part.iterrows(), total=len(df_sess_part)):
#         row['feature_list'] = get_feature_list(row['prev_items_list'], row['next_item'], row['locale'])
#     df_sess_part['feature_list'].to_pickle('df_sess_feature'+str(i)+'.pkl')


# # load df_sess['total_feature_list'] from file by pickle
# for i in range(20):
#     df_sess_part = pd.read_pickle('df_sess_feature'+str(i)+'.pkl')


