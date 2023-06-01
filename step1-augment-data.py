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


# define the function which augment the training data on device
def augment_data(train_data, train_mask):
    # train_data, train_mask = train_data.to(device), train_mask.to(device)
    aug_data, aug_label, aug_mask = torch.tensor([]).to(device), torch.tensor([]).to(device), torch.tensor([]).to(device)

    # count the number of saved samples
    count = 0

    for i in tqdm(range(train_data.shape[0])):
        # append positive samples
        item, mask, label = train_data[i, :, :].to(device), train_mask[i, :].to(device), torch.tensor([1]).to(device)
        aug_data = torch.cat((aug_data, item.unsqueeze(0).to(device)), dim=0)
        aug_mask = torch.cat((aug_mask, mask.unsqueeze(0).to(device)), dim=0)
        aug_label = torch.cat((aug_label, label.unsqueeze(0).to(device)), dim=0).to(device)

    	# generate negative samples
        random_item = train_data[random.randint(0, train_data.shape[0] - 1)].squeeze().to(device)
        while torch.equal(item, random_item):
            random_item = train_data[random.randint(0, train_data.shape[0] - 1)].squeeze().to(device)

        if not torch.equal(item[-1, :], random_item[-1, :]):
            neg_item, neg_mask, neg_label = torch.cat((item[:-1, :], random_item[-1, :].unsqueeze(0)), dim=0), train_mask[i, :], torch.tensor([0])
            aug_data = torch.cat((aug_data, neg_item.unsqueeze(0).to(device)), dim=0)
            aug_mask = torch.cat((aug_mask, neg_mask.unsqueeze(0).to(device)), dim=0)
            aug_label = torch.cat((aug_label, neg_label.unsqueeze(0).to(device)), dim=0)

        # save the augmented data every 100,000 iterations
        if i != 0 and i % 5e4 == 0:
            torch.save(aug_data, 'aug_data_' + str(count) + '.pt')
            torch.save(aug_mask, 'aug_mask_' + str(count) + '.pt')
            torch.save(aug_label, 'aug_label_' + str(count) + '.pt')
            aug_data, aug_label, aug_mask = torch.tensor([]).to(device), torch.tensor([]).to(device), torch.tensor([]).to(device)
            count += 1

    torch.save(aug_data, 'aug_data_' + str(count) + '.pt')
    torch.save(aug_mask, 'aug_mask_' + str(count) + '.pt')
    torch.save(aug_label, 'aug_label_' + str(count) + '.pt')
    return aug_data, aug_mask, aug_label






# define the function which augment the training data on device
# shuffle the data before augmentation
def augment_shuffled_data(train_data, train_mask):
    # train_data, train_mask = train_data.to(device), train_mask.to(device)
    shuffle_index=torch.randperm(train_data.shape[0])
    train_data=train_data[shuffle_index,:,:]
    train_mask=train_mask[shuffle_index,:]
    
    aug_data, aug_label, aug_mask = torch.tensor([]).to(device), torch.tensor([]).to(device), torch.tensor([]).to(device)
    
    # count the number of saved samples
    count = 0
    
    for i in tqdm(range(train_data.shape[0])):
        # append positive samples
        item, mask, label = train_data[i, :, :].to(device), train_mask[i, :].to(device), torch.tensor([1]).to(device)
        aug_data = torch.cat((aug_data, item.unsqueeze(0).to(device)), dim=0)
        aug_mask = torch.cat((aug_mask, mask.unsqueeze(0).to(device)), dim=0)
        aug_label = torch.cat((aug_label, label.unsqueeze(0).to(device)), dim=0).to(device)
    
    	# generate negative samples
        random_item_id = random.randint(0, train_data.shape[0] - 1)
        random_item = train_data[random_item_id].squeeze().to(device)
        while torch.equal(item, random_item):
            random_item_id = random.randint(0, train_data.shape[0] - 1)
            random_item = train_data[random_item_id].squeeze().to(device)

        # get mask length of item and random_item
        item_seq_len = (int)(torch.sum(mask, dtype=torch.int32))
        random_item_seq_len = (int)(torch.sum(train_mask[random_item_id, :], dtype=torch.int32))
        
        if not torch.equal(item[item_seq_len-1, :], random_item[random_item_seq_len-1, :]):
            neg_item, neg_mask, neg_label = torch.cat((item[:item_seq_len-1, :], random_item[random_item_seq_len-1, :].unsqueeze(0)), dim=0), mask, torch.tensor([0])
            # pad the negative item to the same length as the positive item
            neg_item = torch.cat((neg_item, torch.zeros((item.shape[0] - neg_item.shape[0], item.shape[1])).to(device)), dim=0)
        
            aug_data = torch.cat((aug_data, neg_item.unsqueeze(0).to(device)), dim=0)
            aug_mask = torch.cat((aug_mask, neg_mask.unsqueeze(0).to(device)), dim=0)
            aug_label = torch.cat((aug_label, neg_label.unsqueeze(0).to(device)), dim=0)
        
        # save the augmented data every 100,000 iterations
        if i != 0 and i % 5e4 == 0:
            torch.save(aug_data, 'aug_data_' + str(count) + '.pt')
            torch.save(aug_mask, 'aug_mask_' + str(count) + '.pt')
            torch.save(aug_label, 'aug_label_' + str(count) + '.pt')
            aug_data, aug_label, aug_mask = torch.tensor([]).to(device), torch.tensor([]).to(device), torch.tensor([]).to(device)
            count += 1

    torch.save(aug_data, 'aug_data_' + str(count) + '.pt')
    torch.save(aug_mask, 'aug_mask_' + str(count) + '.pt')
    torch.save(aug_label, 'aug_label_' + str(count) + '.pt')

    # load all the augmented data
    train_data, train_mask, train_label = torch.tensor([]), torch.tensor([]), torch.tensor([])
    for i in range(count + 1):
        train_data = torch.cat((train_data, torch.load('aug_data_' + str(i) + '.pt').cpu()), dim=0)
        train_mask = torch.cat((train_mask, torch.load('aug_mask_' + str(i) + '.pt').cpu()), dim=0)
        train_label = torch.cat((train_label, torch.load('aug_label_' + str(i) + '.pt').cpu()), dim=0)
    
    # save the augmented data
    torch.save(train_data, 'final_aug_train_data.pt')
    torch.save(train_mask, 'final_aug_train_mask.pt')
    torch.save(train_label, 'final_aug_train_label.pt')
    return train_data, train_mask, train_label


# ---------------------------- begin to augment data ---------------------------- #

# import train_data
feature_list = None
with open('feature_list.pkl', 'rb') as f:
    feature_list = pickle.load(f)

# convert feature_list to pandas Series
train_data = pd.Series(feature_list).apply(lambda x: np.array(x)).apply(lambda x: torch.from_numpy(x))

# # data analysis

# # generate padding for the train_data on the 2th dimension in (batch_size, seq_len, embedding_dim)
# # get the max seq_len
# maxlen = train_data.apply(lambda x: x.shape[0]).max()
# print(f'maxlen: {maxlen}')

# # plot the distribution of train_data.shape[0]
# train_data.apply(lambda x: x.shape[0]).hist(bins=200)
# plt.savefig('train_data_hist.png')
# plt.close()



# find the 99% percentile of train_data.shape[0]
percentile = train_data.apply(lambda x: x.shape[0]).quantile(0.99)
# use a len larger than the 99% percentile as the maxlen
maxlen = int(percentile) + 1

# delete the train_data with shape[0] larger than maxlen
maxlen = 20
train_data = train_data[train_data.apply(lambda x: x.shape[0]) <= maxlen]

# pad each item of train_data using torch.nn.functional.pad on the 1th dimension with value 0
train_data = train_data.apply(lambda x: F.pad(x, (0, 0, 0, maxlen - x.shape[0]), 'constant', 0))

# convert the train_data to torch.tensor
train_data = torch.stack(tuple(train_data.values))

# save the train_data
torch.save(train_data, 'train_data.pt')

# load the train_data
train_data = torch.load('train_data.pt')



# construct mask for train_data
train_mask = torch.zeros(train_data.shape[0], train_data.shape[1])

for i in tqdm(range(train_data.shape[0])):
    for j in range(train_data.shape[1]):
        if not torch.equal(train_data[i, j, :], torch.zeros(train_data.shape[2], device=device)):
            train_mask[i, j] = 1

# save the train_mask
torch.save(train_mask, 'train_mask.pt')
print(f'train_mask shape: {train_mask.shape}')

# load the train_mask
train_mask = torch.load('train_mask.pt')

# augment train data and mask
train_data, train_mask, train_label = augment_shuffled_data(train_data, train_mask)