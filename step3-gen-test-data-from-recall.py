import warnings
warnings.simplefilter('ignore')

import pickle
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
pd.set_option('display.max_columns', None)
from tqdm.auto import tqdm

import torch
import torch.nn.functional as F

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")


df_prod = pd.read_csv('products_train.csv')
df_prod

df_test = pd.read_csv('sessions_test_task1.csv')
df_test


def str2list(x):
    x = x.replace('[', '').replace(']', '').replace("'", '').replace('\n', ' ').replace('\r', ' ')
    l = [i for i in x.split() if i]
    return l

df_test['prev_items_list'] = df_test['prev_items'].apply(lambda x: str2list(x))

df_test['prev_items_list'].apply(lambda x: len(x)).max() # maxlen: 112


id_num = torch.load('id_num.pt')
locale_onehot = torch.load('locale_onehot.pt')
price_onehot = torch.load('price_onehot.pt')

nontext_features = torch.cat((id_num, locale_onehot, price_onehot), dim=1)

# save feature_tensor to file
torch.save(nontext_features, 'nontext_features.pt')
# # load feature_tensor from file
# nontext_features = torch.load('nontext_features.pt')


# make a map from item to its feature tensor
# to fasten search the corresponding feature tensor
item2feature = {}
for i, row in tqdm(df_prod.iterrows(), total=len(df_prod)):
    # get numpy array of feature tensor feature_tensor[i]
    item2feature[str(row['id']) + ' ' + str(row['locale'])] = nontext_features[i].cpu().numpy()


# get features by id + locale
in_prods = 0
out_prods = 0
def get_feature(item, locale):
    # find feature tensor for feature_tensor
    feature = None
    if (item + ' ' + locale) in item2feature:
        feature = item2feature[item + ' ' + locale]
        global in_prods
        in_prods += 1
    else:
        global out_prods
        out_prods += 1
    return feature


def get_feature_list(prev_items_list, next_item, locale):
    feature_list = []
    # get the feature tensor for prev_items_list
    for item in prev_items_list:
        if get_feature(item, locale) is not None:
            feature_list.append(get_feature(item, locale))
    
    # get the feature tensor for next_item
    if get_feature(next_item, locale) is not None:
        feature_list.append(get_feature(next_item, locale))
    
    return feature_list


# load the result of recall
with open('rule_recall_100_test.pkl', 'rb') as f:
    rule_recall_100_test = pickle.load(f)


# get feature list for each row of df_test and corresponding next_item_prediction
test_final_feature_list = []
next_item_prediction = rule_recall_100_test['next_item_prediction']
for i, row in tqdm(df_test.iterrows(), total=len(df_test)):
    test_feature_list = []
    for item in next_item_prediction[i]:
        my_feature = get_feature_list(row['prev_items_list'], item, row['locale'])
        test_feature_list.append(my_feature)
    test_final_feature_list.append(test_feature_list)


# generate test_data


# convert test_final_feature_list to pandas DataFrame
test_final_feature_list = pd.DataFrame(test_final_feature_list)


# # find the maxlen in the 3rd dimension and the corresponding index
# maxlen = test_final_feature_list.apply(lambda x: x.apply(lambda y: len(y))).max().max()
# maxlen_index = test_final_feature_list.apply(lambda x: x.apply(lambda y: len(y))).max(axis=1).idxmax()


# cut the test_final_feature_list to maxlen in the 3rd dimension
maxlen = 20
test_data = test_final_feature_list.apply(lambda x: x.apply(lambda y: y[: (min(len(y), maxlen)) ]))


# convert test_data to torch.Tensor
test_data = test_data.apply(lambda x: x.apply(lambda y: np.array(y)))

test_data = test_data.apply(lambda x: x.apply(lambda y: torch.from_numpy(y)))


# pad each item of test_data using torch.nn.functional.pad on the 3th dimension with value 0
test_data = test_data.apply(lambda x: x.apply(lambda y: F.pad(y, (0, 0, 0, maxlen - y.shape[0]), 'constant', 0)))


# convert the test_data to torch.tensor
list1 = []
for i in tqdm(range(len(test_data))):
    list2 = []
    for j in range(len(test_data.iloc[i])):
        list2.append(test_data.iloc[i][j])
    list2 = torch.stack(list2)
    list1.append(list2)

test_data = torch.stack(list1)

# save the test_data
torch.save(test_data, 'test_data.pt')

# load the test_data
test_data = torch.load('test_data.pt')

# separate test_data to 20 pieces and save them
# the total length of test_data is 316971
for i in tqdm(range(20)):
    # copy test_data[i*16000: (i+1)*16000] to test_data_i
    test_data_i = test_data[i*16000: (i+1)*16000].clone()
    # save test_data_i
    torch.save(test_data_i, 'test_data_' + str(i) + '.pt')


# # separate test_mask to 20 pieces and save them
# # the total length of test_mask is 316971
# for i in tqdm(range(20)):
#     # copy test_mask[i*16000: (i+1)*16000] to test_mask_i
#     test_mask_i = test_mask[i*16000: (i+1)*16000].clone()
#     # save test_mask_i
#     torch.save(test_mask_i, 'test_mask_' + str(i) + '.pt')