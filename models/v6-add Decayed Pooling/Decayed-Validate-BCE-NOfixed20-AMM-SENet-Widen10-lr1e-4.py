# build a classification model
# the model first needs to use self-attention mechanism to pool the input vectors into one vector with same dimension
# then the pooled vector is fed into SENet
# the output of SENet is then fed into a fully connected layer to get the final output
# the model is trained with cross entropy loss
# the model is evaluated with accuracy
# the model is saved to a file
# the model is loaded from a file


import random
from tqdm import *
import pickle
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.metrics import accuracy_score

import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader
import os
import wandb

run = wandb.init(
    # Set the project where this run will be logged
    project="kddcup-v1",
    # Track hyperparameters and run metadata
    config={
        "learning_rate": 0.001,
        "epochs": 10,
        "batch_size": 256,
        "structure": "nofixed-bce",
    },
    name="Decayed-0.9")

torch.manual_seed(0)

# set the random seed for reproducibility
np.random.seed(100)

# set the device
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

# set the device to be the second GPU if you have two GPUs
device = torch.device('cuda:1' if torch.cuda.is_available() else 'cpu')

# define the target attention layer using Q, K, V with mask
# the input is a 3D tensor with shape (batch_size, seq_len, input_dim)
# the output is a 2D tensor with shape (batch_size, output_dim)
class AttentionLayerWithMask(nn.Module):
    def __init__(self, input_dim, output_dim):
        super(AttentionLayerWithMask, self).__init__()
        self.input_dim = input_dim
        self.output_dim = output_dim
        self.Wq = nn.Linear(input_dim, output_dim)
        self.Wk = nn.Linear(input_dim, output_dim)
        self.Wv = nn.Linear(input_dim, output_dim)
        self.softmax = nn.Softmax(dim=-1)
        self.tanh = nn.Tanh()
        self.dropout = nn.Dropout(p=0.5)
    def forward(self, input, mask):
        # input is a 3D tensor with shape (batch_size, seq_len, input_dim)
        # mask is a 2D tensor with shape (batch_size, seq_len)
        seq_len = torch.ones(mask.shape[0], dtype=torch.int32) * mask.shape[1]
        # use mask to mask out the padded values using 1e-9
        if mask is not None:
            # get the number of 1 from mask
            # the output is a 1D tensor with shape (batch_size) in integer
            seq_len = torch.sum(mask, dim=1, dtype=torch.int32)
        output = torch.Tensor().to(device)
        for i in range(input.shape[0]):
            sqlen = seq_len[i]
            q = self.Wq(input[i, sqlen-1, :].unsqueeze(0))
            k = self.Wk(input[i, :sqlen-1, :])
            v = self.Wv(input[i, :sqlen-1, :])
            qkt = torch.mm(q, k.transpose(0, 1))
            # use softmax to normalize the weights
            qkt = self.softmax(qkt)
            # QK^TV is a 3D tensor with shape (batch_size, seq_len, output_dim)
            # the output is a 3D tensor with shape (batch_size, seq_len, output_dim)
            qktv = torch.mm(qkt, v)
            output = torch.cat((output, qktv), dim=0)   
        # the output is a 2D tensor with shape (batch_size, output_dim)
        output = self.tanh(output)
        # the output is a 2D tensor with shape (batch_size, output_dim)
        output = self.dropout(output)
        return output



# define the average pooling layer similar to the AttentionLayerWithMask
# the input is a 3D tensor with shape (batch_size, seq_len, input_dim)
# the output is a 2D tensor with shape (batch_size, output_dim)
class AveragePoolingLayerWithMask(nn.Module):
    def __init__(self, input_dim, output_dim):
        super(AveragePoolingLayerWithMask, self).__init__()
        self.input_dim = input_dim
        self.output_dim = output_dim
    def forward(self, input, mask):
        # input is a 3D tensor with shape (batch_size, seq_len, input_dim)
        # mask is a 2D tensor with shape (batch_size, seq_len)
        seq_len = torch.ones(mask.shape[0], dtype=torch.int32) * mask.shape[1]
        # use mask to mask out the padded values using 1e-9
        if mask is not None:
            # get the number of 1 from mask
            # the output is a 1D tensor with shape (batch_size) in integer
            seq_len = torch.sum(mask, dim=1, dtype=torch.int32)
        output = torch.Tensor().to(device)
        for i in range(input.shape[0]):
            sqlen = seq_len[i]
            # the output is a 2D tensor with shape (batch_size, output_dim)
            output = torch.cat((output, torch.mean(input[i, :sqlen, :], dim=0).unsqueeze(0)), dim=0)
        return output




# # define a time-decayed average pooling layer similar to the AttentionLayerWithMask
# # the input is a 3D tensor with shape (batch_size, seq_len, input_dim)
# # the output is a 2D tensor with shape (batch_size, output_dim)
# # the pooling weights of one item depends on the distance from it to the last item
# class TimeDecayedAveragePoolingLayerWithMask(nn.Module):
#     def __init__(self, input_dim, output_dim, alpha=0.5):
#         super(TimeDecayedAveragePoolingLayerWithMask, self).__init__()
#         self.input_dim = input_dim
#         self.output_dim = output_dim
#         self.alpha = alpha
#     def forward(self, input, mask):
#         # input is a 3D tensor with shape (batch_size, seq_len, input_dim)
#         # mask is a 2D tensor with shape (batch_size, seq_len)
#         seq_len = torch.ones(mask.shape[0], dtype=torch.int32) * mask.shape[1]
#         # use mask to mask out the padded values using 1e-9
#         if mask is not None:
#             # get the number of 1 from mask
#             # the output is a 1D tensor with shape (batch_size) in integer
#             seq_len = torch.sum(mask, dim=1, dtype=torch.int32)
#         output = torch.Tensor().to(device)
#         for i in range(input.shape[0]):
#             sqlen = seq_len[i]
#             # the output is a 2D tensor with shape (batch_size, output_dim)
#             # here we use the time-decayed average pooling
#             # the weight of each item is alpha ^ (distance to the last item) and the weight of the last item is 1
#             weight = torch.Tensor().to(device)
#             for j in range(sqlen):
#                 weight = torch.cat((weight, torch.Tensor([self.alpha ** (sqlen - j - 1)]).to(device)), dim=0)
#             weight = weight / torch.sum(weight)
#             # the output is a 2D tensor with shape (batch_size, output_dim)
#             output = torch.cat((output, torch.mm(weight.unsqueeze(0), input[i, :sqlen, :])), dim=0)
#         return output





# define a time-decayed average pooling layer similar to the AttentionLayerWithMask
# the input is a 3D tensor with shape (batch_size, seq_len, input_dim)
# the output is a 2D tensor with shape (batch_size, output_dim)
# the pooling weights of one item depends on the distance from it to the last item
class TimeDecayedAveragePoolingLayerWithMask(nn.Module):
    def __init__(self, input_dim, output_dim, alpha=0.5):
        super(TimeDecayedAveragePoolingLayerWithMask, self).__init__()
        self.input_dim = input_dim
        self.output_dim = output_dim
        self.alpha = alpha
        self.weight = torch.Tensor().to(device)
        self.maxlen = 20
        # here we use the time-decayed average pooling
            # the weight of each item is alpha ^ (distance to the last item) and the weight of the last item is 1
        for i in range(self.maxlen):
            self.weight = torch.cat((self.weight, torch.Tensor([self.alpha ** (self.maxlen - i - 1)]).to(device)), dim=0)
    def forward(self, input, mask):
        # input is a 3D tensor with shape (batch_size, seq_len, input_dim)
        # mask is a 2D tensor with shape (batch_size, seq_len)
        seq_len = torch.ones(mask.shape[0], dtype=torch.int32) * mask.shape[1]
        # use mask to mask out the padded values using 1e-9
        if mask is not None:
            # get the number of 1 from mask
            # the output is a 1D tensor with shape (batch_size) in integer
            seq_len = torch.sum(mask, dim=1, dtype=torch.int32)
        output = torch.Tensor().to(device)
        for i in range(input.shape[0]):
            sqlen = seq_len[i]
            # the output is a 2D tensor with shape (batch_size, output_dim)
            # get the weight of each item, i.e., last item has weight 1, second last item has weight alpha, etc.
            weight = self.weight[self.maxlen - sqlen:]
            weight = weight / torch.sum(weight)
            # the output is a 2D tensor with shape (batch_size, output_dim)
            output = torch.cat((output, torch.mm(weight.unsqueeze(0), input[i, :sqlen, :])), dim=0)
        return output


















# define the max pooling layer similar to the AttentionLayerWithMask
# the input is a 3D tensor with shape (batch_size, seq_len, input_dim)
# the output is a 2D tensor with shape (batch_size, output_dim)
class MaxPoolingLayerWithMask(nn.Module):
    def __init__(self, input_dim, output_dim):
        super(MaxPoolingLayerWithMask, self).__init__()
        self.input_dim = input_dim
        self.output_dim = output_dim
    def forward(self, input, mask):
        # input is a 3D tensor with shape (batch_size, seq_len, input_dim)
        # mask is a 2D tensor with shape (batch_size, seq_len)
        seq_len = torch.ones(mask.shape[0], dtype=torch.int32) * mask.shape[1]
        # use mask to mask out the padded values using 1e-9
        if mask is not None:
            # get the number of 1 from mask
            # the output is a 1D tensor with shape (batch_size) in integer
            seq_len = torch.sum(mask, dim=1, dtype=torch.int32)
        output = torch.Tensor().to(device)
        for i in range(input.shape[0]):
            sqlen = seq_len[i]
            # the output is a 2D tensor with shape (batch_size, output_dim)
            max_output = torch.max(input[i, :sqlen, :], dim=0)[0]
            # clip max_output to avoid nan using torch.clamp
            max_output = torch.clamp(max_output, min=1e-9, max=1e9)
            output = torch.cat((output, max_output.unsqueeze(0)), dim=0)
        return output


# define the min pooling layer similar to the AttentionLayerWithMask
# the input is a 3D tensor with shape (batch_size, seq_len, input_dim)
# the output is a 2D tensor with shape (batch_size, output_dim)
class MinPoolingLayerWithMask(nn.Module):
    def __init__(self, input_dim, output_dim):
        super(MinPoolingLayerWithMask, self).__init__()
        self.input_dim = input_dim
        self.output_dim = output_dim
    def forward(self, input, mask):
        # input is a 3D tensor with shape (batch_size, seq_len, input_dim)
        # mask is a 2D tensor with shape (batch_size, seq_len)
        seq_len = torch.ones(mask.shape[0], dtype=torch.int32) * mask.shape[1]
        # use mask to mask out the padded values using 1e-9
        if mask is not None:
            # get the number of 1 from mask
            # the output is a 1D tensor with shape (batch_size) in integer
            seq_len = torch.sum(mask, dim=1, dtype=torch.int32)
        output = torch.Tensor().to(device)
        for i in range(input.shape[0]):
            sqlen = seq_len[i]
            # the output is a 2D tensor with shape (batch_size, output_dim)
            min_output = torch.min(input[i, :sqlen, :], dim=0)[0]
            # clip max_output to avoid nan using torch.clamp
            min_output = torch.clamp(min_output, min=1e-9, max=1e9)
            output = torch.cat((output, min_output.unsqueeze(0)), dim=0)
        return output



# define the SENet
# the input is a 2D tensor with shape (batch_size, input_dim)
# the output is a 2D tensor with shape (batch_size, output_dim) with output_dim = input_dim
class SENet(nn.Module):
    def __init__(self, input_dim):
        super(SENet, self).__init__()
        self.input_dim = input_dim
        self.fc1 = nn.Linear(12, 3)
        self.fc2 = nn.Linear(3, 12)
        self.sigmoid = nn.Sigmoid()
        self.dropout = nn.Dropout(p=0.5)
    def forward(self, input):
        # seperate the input according to the feature dimension
        # prevs avg
        prevs_id_num_avg = input[:, :20]
        prevs_locale_avg = input[:, 20:26]
        prevs_price_avg = input[:, 26:47]
        # prevs max
        prevs_id_num_max = input[:, 47:67]
        prevs_locale_max = input[:, 67:73]
        prevs_price_max = input[:, 73:94]
        # prevs min
        prevs_id_num_min = input[:, 94:114]
        prevs_locale_min = input[:, 114:120]
        prevs_price_min = input[:, 120:141]
        # next
        next_id_num = input[:, 141:161]
        next_locale = input[:, 161:167]
        next_price = input[:, 167:188]
        # concatenate the three 2D tensors into a 2D tensor with shape (batch_size, 3)
        se_weights = torch.stack((torch.mean(prevs_id_num_avg, dim=1), torch.mean(prevs_locale_avg, dim=1), torch.mean(prevs_price_avg, dim=1),
                                  torch.mean(prevs_id_num_max, dim=1), torch.mean(prevs_locale_max, dim=1), torch.mean(prevs_price_max, dim=1),
                                  torch.mean(prevs_id_num_min, dim=1), torch.mean(prevs_locale_min, dim=1), torch.mean(prevs_price_min, dim=1),
                                  torch.mean(next_id_num, dim=1), torch.mean(next_locale, dim=1), torch.mean(next_price, dim=1)), dim=1).to(device)
        # use a fully connected layer with relu activation to shrink the 2D tensor into a 2D tensor with shape (batch_size, input_dim // 4)
        se_weights = self.fc1(se_weights)
        se_weights = F.relu(se_weights)
        # use a fully connected layer with sigmoid activation to expand the 2D tensor into a 2D tensor with shape (batch_size, input_dim)
        se_weights = self.fc2(se_weights)
        se_weights = self.sigmoid(se_weights)
        # row-wise multiply se_weights with input
        # multiply the weights with the prevs and next use broadcasting to do the multiplication
        # prevs avg
        output = torch.mul(se_weights[:, 0].unsqueeze(1), prevs_id_num_avg)
        output = torch.cat((output, se_weights[:, 1].unsqueeze(1) * prevs_locale_avg), dim=1)
        output = torch.cat((output, se_weights[:, 2].unsqueeze(1) * prevs_price_avg), dim=1)
        # prevs max
        output = torch.cat((output, se_weights[:, 3].unsqueeze(1) * prevs_id_num_max), dim=1)
        output = torch.cat((output, se_weights[:, 4].unsqueeze(1) * prevs_locale_max), dim=1)
        output = torch.cat((output, se_weights[:, 5].unsqueeze(1) * prevs_price_max), dim=1)
        # prevs min
        output = torch.cat((output, se_weights[:, 6].unsqueeze(1) * prevs_id_num_min), dim=1)
        output = torch.cat((output, se_weights[:, 7].unsqueeze(1) * prevs_locale_min), dim=1)
        output = torch.cat((output, se_weights[:, 8].unsqueeze(1) * prevs_price_min), dim=1)
        # next
        output = torch.cat((output, se_weights[:, 9].unsqueeze(1) * next_id_num), dim=1)
        output = torch.cat((output, se_weights[:, 10].unsqueeze(1) * next_locale), dim=1)
        output = torch.cat((output, se_weights[:, 11].unsqueeze(1) * next_price), dim=1)
        # the output is a 2D tensor with shape (batch_size, input_dim)
        output = self.dropout(output)
        return output




# write a residual block
# the input is a 2D tensor with shape (batch_size, input_dim)
# the output is a 2D tensor with shape (batch_size, input_dim)
class ResidualBlock(nn.Module):
    def __init__(self, input_dim):
        super(ResidualBlock, self).__init__()
        self.input_dim = input_dim
        self.fc1 = nn.Linear(input_dim, input_dim)
        self.fc2 = nn.Linear(input_dim, input_dim)
        self.relu = nn.ReLU()
        self.dropout = nn.Dropout(p=0.5)
    def forward(self, input):
        # the output is a 2D tensor with shape (batch_size, input_dim)
        output = self.fc1(input)
        output = self.relu(output)
        output = self.dropout(output)
        output = self.fc2(output)
        output = self.relu(output)
        output = self.dropout(output)
        # the output is a 2D tensor with shape (batch_size, input_dim)
        output = output + input
        return output


# write a new residual block
# the input is a 2D tensor with shape (batch_size, input_dim)
# the output is a 2D tensor with shape (batch_size, input_dim)
class ResidualBlockWiden(nn.Module):
    def __init__(self, input_dim):
        super(ResidualBlockWiden, self).__init__()
        self.input_dim = input_dim
        self.fc1 = nn.Linear(input_dim, input_dim * 4)
        self.fc2 = nn.Linear(input_dim * 4, input_dim)
        self.relu = nn.ReLU()
        self.dropout = nn.Dropout(p=0.5)
    def forward(self, input):
        # the output is a 2D tensor with shape (batch_size, input_dim)
        output = self.fc1(input)
        output = self.relu(output)
        output = self.dropout(output)
        output = self.fc2(output)
        output = self.relu(output)
        output = self.dropout(output)
        # the output is a 2D tensor with shape (batch_size, input_dim)
        output = output + input
        return output



# define the model
# the input is a 3D tensor with shape (batch_size, seq_len, input_dim)
# the output is a tensor with shape (batch_size, 2, 200)
# which represents two (batch_size, 200) vectors for the prev_items(session) and the next_item
class Model(nn.Module):
    def __init__(self, input_dim, output_dim, vocab_size):
        super(Model, self).__init__()
        self.input_dim = input_dim
        self.output_dim = output_dim
        # This is original attention
        # self.attention = AttentionLayer(input_dim, input_dim)
        # We use target attention in recommendation system
        # ---
        # self.attention = AttentionLayerWithMask(input_dim, input_dim)
        self.td_avgpool = TimeDecayedAveragePoolingLayerWithMask(input_dim, input_dim, alpha=0.9)
        self.maxpool = MaxPoolingLayerWithMask(input_dim, input_dim)
        self.minpool = MinPoolingLayerWithMask(input_dim, input_dim)
        self.se = SENet(input_dim)
        # use residual blocks
        self.residual_block1 = ResidualBlockWiden(4 * input_dim)
        self.residual_block2 = ResidualBlockWiden(4 * input_dim)
        self.residual_block3 = ResidualBlockWiden(4 * input_dim)
        self.residual_block4 = ResidualBlockWiden(4 * input_dim)
        self.residual_block5 = ResidualBlockWiden(4 * input_dim)
        self.residual_block6 = ResidualBlockWiden(4 * input_dim)
        self.residual_block7 = ResidualBlockWiden(4 * input_dim)
        self.residual_block8 = ResidualBlockWiden(4 * input_dim)
        self.residual_block9 = ResidualBlockWiden(4 * input_dim)
        self.residual_block10 = ResidualBlockWiden(4 * input_dim)
        # get the final 1-dimensional output
        self.fc = nn.Linear(4 * input_dim, output_dim)
        # use nn.Embedding to get the embedding of the input with 20 dimensions
        self.embed = nn.Embedding(vocab_size, 20)
    def forward(self, input, mask):
        # input is a 3D tensor with shape (batch_size, seq_len, input_dim)
        # the output is a 2D tensor with shape (batch_size, 1)
        # the output is a probability between 0 and 1
        # get the first dimension of the input
        id_num = input[:, :, 0].long()
        # use nn.Embedding to get the embedding of the input with 20 dimensions
        id_embeddings = self.embed(id_num)
        # replace id_num with id_embeddings in the input
        input = torch.cat((id_embeddings, input[:, :, 1:]), dim=2)  
        # apply the attention layer to the input
        # the output is a 2D tensor with shape (batch_size, input_dim)
        # next item & prev items
        next_item_output = input[:, -1, :].squeeze()
        prev_items_output_td_avgpool = self.td_avgpool(input, mask)
        prev_items_output_maxpool = self.maxpool(input, mask)
        prev_items_output_minpool = self.minpool(input, mask)
        output = torch.cat((prev_items_output_td_avgpool, prev_items_output_maxpool, prev_items_output_minpool, next_item_output), dim=1)
        # apply the SENet to the output of the attention layer
        # the output is a 2D tensor with shape (batch_size, input_dim)
        output = self.se(output)
        # apply a series of residual blocks to the output of the SENet
        # the output is a 2D tensor with shape (batch_size, input_dim)
        output = self.residual_block1(output)
        output = self.residual_block2(output)
        output = self.residual_block3(output)
        output = self.residual_block4(output)
        output = self.residual_block5(output)
        output = self.residual_block6(output)
        output = self.residual_block7(output)
        output = self.residual_block8(output)
        output = self.residual_block9(output)
        output = self.residual_block10(output)
        # the output is a 2D tensor with shape (batch_size, 1)
        output = self.fc(output)
        # This is for classification task
        # use softmax to convert the output to a probability
        # the output is a 2D tensor with shape (batch_size, 2)
        # output = F.softmax(output, dim=1)
        return output



# define the dataloader
class MyDataset(Dataset):
    def __init__(self, data, mask, label):
        self.data = data
        self.mask = mask
        self.label = label
    def __getitem__(self, index):
        return self.data[index], self.mask[index], self.label[index]
    def __len__(self):
        return len(self.data)




# load the validation data
df_validate = pd.read_pickle('rule_recall_100_validate.pkl')
df_validate


# write a function to calculate the MRR score
def calculate_mrr(df):
    mrr = 0
    for _, row in tqdm(df.iterrows(), total=len(df)):
        if row['next_item'] in row['next_item_prediction']:
            mrr_score = 1 / (row['next_item_prediction'].index(row['next_item']) + 1)
            mrr += mrr_score
    mrr /= len(df)
    return mrr



df_prod = pd.read_csv('products_train.csv')
df_prod

nontext_features = torch.load('nontext_features.pt')

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


def str2list(x):
    x = x.replace('[', '').replace(']', '').replace("'", '').replace('\n', ' ').replace('\r', ' ')
    l = [i for i in x.split() if i]
    return l


def get_validation_data(df_validate):
    # get feature list for each row of df_validate and corresponding next_item_prediction
    validate_feature_list = []
    df_validate['prev_items_list'] = df_validate['prev_items'].apply(lambda x: str2list(x))
    next_item_prediction = df_validate['next_item_prediction']
    for i, row in tqdm(df_validate.iterrows(), total=len(df_validate)):
        test_feature_list = []
        for item in next_item_prediction[i]:
            my_feature = get_feature_list(row['prev_items_list'], item, row['locale'])
            test_feature_list.append(my_feature)
        validate_feature_list.append(test_feature_list)
    # convert validate_feature_list to pandas DataFrame
    validate_feature_list = pd.DataFrame(validate_feature_list)
    # cut the validate_feature_list to maxlen in the 3rd dimension
    maxlen = 20
    final_validate_data = validate_feature_list.apply(lambda x: x.apply(lambda y: y[: (min(len(y), maxlen)) ]))
    # convert final_validate_data to torch.Tensor
    final_validate_data = final_validate_data.apply(lambda x: x.apply(lambda y: np.array(y)))
    final_validate_data = final_validate_data.apply(lambda x: x.apply(lambda y: torch.from_numpy(y)))
    # pad each item of final_validate_data using torch.nn.functional.pad on the 3th dimension with value 0
    final_validate_data = final_validate_data.apply(lambda x: x.apply(lambda y: F.pad(y, (0, 0, 0, maxlen - y.shape[0]), 'constant', 0)))
    # convert the final_validate_data to torch.tensor
    list1 = []
    for i in tqdm(range(len(final_validate_data))):
        list2 = []
        for j in range(len(final_validate_data.iloc[i])):
            list2.append(final_validate_data.iloc[i][j])
        list2 = torch.stack(list2)
        list1.append(list2)
    final_validate_data = torch.stack(list1)    
    # construct mask for final_validate_data
    final_validate_mask = torch.zeros(final_validate_data.shape[0], final_validate_data.shape[1], final_validate_data.shape[2])
    for i in tqdm(range(final_validate_data.shape[0])):
        for j in range(final_validate_data.shape[1]):
            for k in range(final_validate_data.shape[2]):
                if not torch.equal(final_validate_data[i, j, k, :], torch.zeros(final_validate_data.shape[3])):
                    final_validate_mask[i, j, k] = 1
    return final_validate_data, final_validate_mask



final_validate_data, final_validate_mask = get_validation_data(df_validate)



# save final_validate_data and final_validate_mask
torch.save(final_validate_data, 'final_validate_data.pt')
torch.save(final_validate_mask, 'final_validate_mask.pt')
# # load final_validate_data and final_validate_mask
# final_validate_data = torch.load('final_validate_data.pt')
# final_validate_mask = torch.load('final_validate_mask.pt')




# define the training function
def train(model, data, mask, label, optimizer, criterion, df_validate):
    # mrr = calculate_mrr(df_validate)
    # wandb.log({'mrr': mrr})
    # set the model to training mode
    model.train()
    dataset = MyDataset(data, mask, label)
    dataloader = DataLoader(dataset, batch_size=256, shuffle=True)
    cnt = 0
    for data_batch, mask_batch, label_batch in tqdm(dataloader):
        cnt += 1
        # move the data to the device
        data_batch, mask_batch, label_batch = data_batch.to(device), mask_batch.to(device), label_batch.to(device)
        # compute the output
        output = model(data_batch, mask_batch)
        # compute the loss
        m = nn.Sigmoid()
        loss = criterion(m(output), label_batch)
        # compute the gradient
        optimizer.zero_grad()
        loss.backward()
        # update the parameters
        optimizer.step()
        # compute the accuracy
        pred_label = torch.round(m(output))
        acc = accuracy_score(label_batch.cpu().numpy(), pred_label.cpu().detach().numpy())
        wandb.log({'loss': loss.item(), 'acc': acc})
        if cnt % 3000 == 0:
            # deep copy the df_validate
            df_validate_copy = pd.DataFrame(df_validate.values, index=df_validate.index, columns=df_validate.columns).reset_index(drop=True)
            # test the model on the validation data
            model.eval()
            m = nn.Sigmoid()
            pred_label = []
            with torch.no_grad():
                for i in range(final_validate_data.shape[1]):
                    data_batch_i = final_validate_data[:, i, :, :].to(device)
                    mask_batch_i = final_validate_mask[:, i, :].to(device)
                    # shape of output: (batch_size, 2)
                    output = model(data_batch_i, mask_batch_i)
                    # shape of pred_label_i: (batch_size, )
                    output = m(output)
                    if i == 0:
                        pred_label_i = output
                    else:
                        pred_label_i = torch.cat((pred_label_i, output), dim=1)
                pred_label.append(pred_label_i)
            pred_label = torch.cat(pred_label, dim=0)
            # handle the pred_label            
            # generate the indices for sorting for each row
            indices = torch.arange(pred_label.size(1)).repeat(pred_label.size(0), 1).to(device)
            # for each row, combine the tensor and indices into a tuple
            indexed_tensor = [list(zip(pred_label[i, :], indices[i, :])) for i in range(pred_label.size(0))]
            # sort each row of indexed_tensor by the descending order of first element in each tuple
            # and then by increasing order of the second element
            for i in tqdm(range(len(indexed_tensor))):
                indexed_tensor[i] = sorted(indexed_tensor[i], key=lambda x: (-x[0], x[1]))
            # get the indices of the sorted_tensor
            sorted_tensor = [[indexed_tensor[i][j][1] for j in range(len(indexed_tensor[i]))] for i in range(len(indexed_tensor))]
            # convert the elements in sorted_tensor to the int type
            sorted_tensor = [list(map(int, sorted_tensor[i])) for i in range(len(sorted_tensor))]
            # # ---
            # # OR extract the [0,1,2] to be the first three elements in each row of sorted_tensor and then append the rest different elements in this row
            # first_list = [0,1,2]
            # for i in tqdm(range(len(sorted_tensor))):
            #     sorted_tensor[i] = first_list + [j for j in sorted_tensor[i] if j not in first_list]
            # # ---
            # get the value of the corresponding indices in rule_recall_100_test['next_item_prediction']
            sorted_recall_test = [[df_validate_copy['next_item_prediction'][i][j] for j in sorted_tensor[i]] for i in range(len(sorted_tensor))]
            # replace the first elements in rule_recall_100_test['next_item_prediction'] with the sorted_recall_test
            df_validate_copy['next_item_prediction'][:len(sorted_recall_test)] = sorted_recall_test
            # calculate the MRR score and log it into wandb
            mrr = calculate_mrr(df_validate_copy)
            wandb.log({'mrr': mrr})
            print(f'mrr: {mrr}')
            model.train()
    return 





# START TRAINING

# load all the training data
train_data = torch.load('final_aug_train_data.pt')
train_mask = torch.load('final_aug_train_mask.pt')
train_label = torch.load('final_aug_train_label.pt')

vocab_size=(int)(torch.max(train_data[:, :, 0])) + 1


# create the model
# input_dim = train_data.shape[2] + (20 - 1)
# because id is converted to id_embedding, so the input_dim should be 47
model = Model( input_dim=47, output_dim=1, vocab_size=vocab_size ) 
model.to(device)


# count the number of parameters of the model
num_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
num_params

# count the number of trainable parameters of SENet layer
num_params_se = sum(p.numel() for p in model.se.parameters() if p.requires_grad)


# criterion = nn.CrossEntropyLoss()
criterion = nn.BCELoss()
optimizer = optim.Adam(model.parameters(), lr=1e-4)

for epoch in range(10):
    print('epoch: {}'.format(epoch+1))
    # train the model using the training set
    train(model, train_data, train_mask, train_label, optimizer, criterion, df_validate)
    # save the model
    torch.save(model.state_dict(), 'decayed_09_bce_nofixed_senet.pth')
    # save the whole model
    torch.save(model, 'decayed_09_bce_nofixed_senet_model.pth')

# # load the model
# model.load_state_dict(torch.load('bce_nofixed_senet_model.pth'))

# # load the whole model
# model = torch.load('bce_nofixed_senet_model.pth')
