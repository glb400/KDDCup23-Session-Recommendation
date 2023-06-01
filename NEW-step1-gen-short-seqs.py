import torch
import torch.nn.functional as F

import random
from tqdm import *
import pickle
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt


# set the random seed for reproducibility
np.random.seed(100)

# set the device
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')



# load df_sess['feature_list'].values from file by pickle
feature_list = None
with open('feature_list.pkl', 'rb') as f:
    feature_list = pickle.load(f)


# convert feature_list to pandas Series
train_data = pd.Series(feature_list).apply(lambda x: np.array(x)).apply(lambda x: torch.from_numpy(x))


maxlen = 5
# cut the train_data with shape[0] larger than maxlen using a sliding window
# train_data = train_data.apply(lambda x: x[: (min(len(x), maxlen))])
train_data_sliding_window = []
for i in tqdm(range(len(train_data))):
    if len(train_data[i]) <= maxlen:
        train_data_sliding_window.append(train_data[i])
        continue
    else:
        for j in range(len(train_data[i]) - maxlen + 1):
            train_data_sliding_window.append(train_data[i][j: j + maxlen])



# convert train_data_sliding_window to pandas Series
train_data = pd.Series(train_data_sliding_window)

# pad each item of train_data using torch.nn.functional.pad on the 1th dimension with value 0
train_data = train_data.apply(lambda x: F.pad(x, (0, 0, 0, maxlen - x.shape[0]), 'constant', 0))

# convert the train_data to torch.tensor
train_data = torch.stack(tuple(train_data.values)).to(device)

# save the train_data
torch.save(train_data.to(device), 'train_data_short_seq.pt')

# load the train_data
train_data = torch.load('train_data_short_seq.pt')



# construct mask for train_data
train_mask = torch.zeros(train_data.shape[0], train_data.shape[1]).to(device)

for i in tqdm(range(train_data.shape[0])):
    for j in range(train_data.shape[1]):
        if not torch.equal(train_data[i, j, :], torch.zeros(train_data.shape[2], device=device)):
            train_mask[i, j] = 1


# save the train_mask
torch.save(train_mask, 'train_mask_short_seq.pt')
print(f'train_mask_short_seq shape: {train_mask.shape}')

# load the train_mask
train_mask = torch.load('train_mask_short_seq.pt')



# we will generate labels and negative samples in the model








