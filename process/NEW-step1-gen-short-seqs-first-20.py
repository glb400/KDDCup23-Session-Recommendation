import random
from tqdm import *
import pickle
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

import torch
import torch.nn.functional as F


# set the random seed for reproducibility
np.random.seed(100)

# set the device
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')


# ----- Generate train_data ----- # 


# load df_sess['feature_list'].values from file by pickle
moco_feature_list = None
with open('moco_feature_list.pkl', 'rb') as f:
    moco_feature_list = pickle.load(f)


# seperate moco_feature_list into 4 parts: prev_items_feature_list / next_item_feature_list / next_item_prediction_feature_list / samples_feature_list
prev_items_feature_list = moco_feature_list['prev_items_feature_list']
next_item_feature_list = moco_feature_list['next_item_feature_list']
next_item_prediction_feature_list = moco_feature_list['next_item_prediction_feature_list']
samples_feature_list = moco_feature_list['samples_feature_list']



# step1: generate train_data from prev_items_feature_list
# convert feature_list to pandas Series
train_data = pd.Series(prev_items_feature_list).apply(lambda x: np.array(x)).apply(lambda x: torch.from_numpy(x))

# define the maxlen of prev items & pad each item of train_data using torch.nn.functional.pad on the 1th dimension with value 0
maxlen = 20
train_data = train_data.apply(lambda x: F.pad(x, (0, 0, 0, maxlen - x.shape[0]), 'constant', 0))

# convert the train_data to torch.Tensor
train_data = torch.stack(tuple(train_data.values)).to(device)

# save the train_data
torch.save(train_data.to(device), 'moco_train_data.pt')
# # load the train_data
# train_data = torch.load('moco_train_data.pt')



# step2: construct train_mask for train_data
train_mask = torch.zeros(train_data.shape[0], train_data.shape[1]).to(device)

for i in tqdm(range(train_data.shape[0])):
    for j in range(train_data.shape[1]):
        if not torch.equal(train_data[i, j, :], torch.zeros(train_data.shape[2], device=device)):
            train_mask[i, j] = 1

# save the train_mask
torch.save(train_mask, 'moco_train_mask.pt')
print(f'moco_train_mask shape: {train_mask.shape}')
# # load the train_mask
# train_mask = torch.load('moco_train_mask.pt')



# step3: generate train_samples from samples_feature_list





















