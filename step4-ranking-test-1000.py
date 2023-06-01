# build a classification model with inputs that have 1538 dimensions using pytorch
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


import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader
import os
import wandb


run = wandb.init(
    # Set the project where this run will be logged
    project="kddcup-test",
    # Track hyperparameters and run metadata
    config={
        "batch_size": 256,
        "structure": "att-senet-fc",
    })


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



# define the SENet
# the input is a 2D tensor with shape (batch_size, input_dim)
# the output is a 2D tensor with shape (batch_size, output_dim) with output_dim = input_dim
class SENet(nn.Module):
    def __init__(self, input_dim):
        super(SENet, self).__init__()
        self.input_dim = input_dim
        self.fc1 = nn.Linear(6, 2)
        self.fc2 = nn.Linear(2, 6)
        self.sigmoid = nn.Sigmoid()
        self.dropout = nn.Dropout(p=0.5)
    def forward(self, input):
        # seperate the input according to the feature dimension
        prevs_id_num = input[:, :20]
        prevs_locale = input[:, 20:26]
        prevs_price = input[:, 26:47]
        next_id_num = input[:, 47:67]
        next_locale = input[:, 67:73]
        next_price = input[:, 73:94]
        # concatenate the three 2D tensors into a 2D tensor with shape (batch_size, 3)
        se_weights = torch.stack((torch.mean(prevs_id_num, dim=1), torch.mean(prevs_locale, dim=1), torch.mean(prevs_price, dim=1),
                                  torch.mean(next_id_num, dim=1), torch.mean(next_locale, dim=1), torch.mean(next_price, dim=1)), dim=1).to(device)
        # use a fully connected layer with relu activation to shrink the 2D tensor into a 2D tensor with shape (batch_size, input_dim // 16)
        se_weights = self.fc1(se_weights)
        se_weights = F.relu(se_weights)
        # use a fully connected layer with sigmoid activation to expand the 2D tensor into a 2D tensor with shape (batch_size, input_dim)
        se_weights = self.fc2(se_weights)
        se_weights = self.sigmoid(se_weights)
        # row-wise multiply se_weights with input
        # multiply the weights with the prevs and next use broadcasting to do the multiplication
        output = torch.mul(se_weights[:, 0].unsqueeze(1), prevs_id_num)
        output = torch.cat((output, se_weights[:, 1].unsqueeze(1) * prevs_locale), dim=1)
        output = torch.cat((output, se_weights[:, 2].unsqueeze(1) * prevs_price), dim=1)
        output = torch.cat((output, se_weights[:, 3].unsqueeze(1) * next_id_num), dim=1)
        output = torch.cat((output, se_weights[:, 4].unsqueeze(1) * next_locale), dim=1)
        output = torch.cat((output, se_weights[:, 5].unsqueeze(1) * next_price), dim=1)
        # the output is a 2D tensor with shape (batch_size, input_dim)
        output = self.dropout(output)
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
        self.attention = AttentionLayerWithMask(input_dim, input_dim)
        self.se = SENet(input_dim)
        # reduce the dimension of the output of SENet to 1/2 of the input dimension
        self.fc1 = nn.Linear(2 * input_dim, input_dim)
        # reduce the dimension of the output of SENet to 1/4 of the input dimension
        self.fc2 = nn.Linear(input_dim, input_dim // 2)
        # reduce the dimension of the output of SENet to 1/8 of the input dimension
        self.fc3 = nn.Linear(input_dim // 2, input_dim // 4)
        # get the final 1-dimensional output
        self.fc4 = nn.Linear(input_dim // 4, output_dim)
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
        prev_items_output = self.attention(input, mask)
        output = torch.cat((prev_items_output, next_item_output), dim=1)
        # apply the SENet to the output of the attention layer
        # the output is a 2D tensor with shape (batch_size, input_dim)
        output = self.se(output)
        # apply a fully connected layer with relu activation to the output of the SENet
        # the output is a 2D tensor with shape (batch_size, input_dim // 2)
        output = self.fc1(output)
        output = F.relu(output)
        # the output is a 2D tensor with shape (batch_size, input_dim // 4)
        output = self.fc2(output)
        output = F.relu(output)
        # the output is a 2D tensor with shape (batch_size, input_dim // 8)
        output = self.fc3(output)
        output = F.relu(output)
        # the output is a 2D tensor with shape (batch_size, 1)
        output = self.fc4(output)
        # This is for classification task
        # use softmax to convert the output to a probability
        # the output is a 2D tensor with shape (batch_size, 2)
        output = F.softmax(output, dim=1)
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


# START EVALUATION

# set the random seed for reproducibility
np.random.seed(100)

# set the device
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

# load the test_data
test_data = torch.load('test_data_0.pt')

# # construct mask for test_data
# test_mask = torch.zeros(test_data.shape[0], test_data.shape[1], test_data.shape[2])
# for i in tqdm(range(test_data.shape[0])):
#     for j in range(test_data.shape[1]):
#         for k in range(test_data.shape[2]):
#             if not torch.equal(test_data[i, j, k, :], torch.zeros(test_data.shape[3])):
#                 test_mask[i, j, k] = 1

# # save the test_mask
# torch.save(test_mask, 'test_mask_0.pt')

# load the test_mask
test_mask = torch.load('test_mask_0.pt')



# (int)(torch.max(train_data[:, :, 0])) + 1
vocab_size = 1410675


# create the model
# input_dim = test_data.shape[2] + (20 - 1)
# because id is converted to id_embedding, so the input_dim should be 47
model = Model( input_dim=47, output_dim=2, vocab_size=vocab_size ) 
model.to(device)

# load the model
# model.load_state_dict(torch.load('fixed_senet_model.pth'))
model = torch.load('nofixed_senet_model.pth')
model.to(device)


# set the model to evaluation mode
model.eval()
dataset = MyDataset(test_data, test_mask, torch.zeros(test_data.shape[0]))
dataloader = DataLoader(dataset, batch_size=256, shuffle=False)



# # want to get the predictions for the last item in each session
# # the shape of pred_label should be (batch_size, data.shape[1])
# pred_label = []
# with torch.no_grad():
#     for data_batch, mask_batch, _ in tqdm(dataloader):
#         for i in range(test_data.shape[1]):
#             data_batch_i = data_batch[:, i, :, :].to(device)
#             mask_batch_i = mask_batch[:, i, :].to(device)
#             # shape of output: (batch_size, 2)
#             output = model(data_batch_i, mask_batch_i)
#             # shape of pred_label_i: (batch_size, )
#             if i == 0:
#                 pred_label_i = output[:, 1].unsqueeze(1)
#             else:
#                 pred_label_i = torch.cat((pred_label_i, output[:, 1].unsqueeze(1)), dim=1)
#             # # if there is any 1 in pred_label_i, then the session is predicted as 1
#             # # then print the session id
#             # if torch.sum(pred_label_i) > 1:
#             #     print(f'i & sum: {i} \ {torch.sum(pred_label_i)}')
#             #     print("---")
#         pred_label.append(pred_label_i)
#         if i > 0:
#             break



# BCE version
# want to get the predictions for the last item in each session
# the shape of pred_label should be (batch_size, data.shape[1])
m = nn.Sigmoid()
pred_label = []
with torch.no_grad():
    for data_batch, mask_batch, _ in tqdm(dataloader):
        for i in range(test_data.shape[1]):
            data_batch_i = data_batch[:, i, :, :].to(device)
            mask_batch_i = mask_batch[:, i, :].to(device)
            # shape of output: (batch_size, 2)
            output = model(data_batch_i, mask_batch_i)
            # shape of pred_label_i: (batch_size, )
            output = m(output)
            if i == 0:
                pred_label_i = output
            else:
                pred_label_i = torch.cat((pred_label_i, output), dim=1)
        pred_label.append(pred_label_i)


# convert pred_label from a list to a 1D array
pred_label = torch.cat(pred_label, dim=0)




# # normalize each row of pred_label in [0, 1]
# pred_label = pred_label - torch.min(pred_label, dim=1)[0].unsqueeze(1)
# pred_label = pred_label / torch.max(pred_label, dim=1)[0].unsqueeze(1)
# # convert nan in pred_label to 1
# pred_label[pred_label != pred_label] = 1




# save the pred_label
torch.save(pred_label, 'pred_label_0.pt')
# # load the pred_label
# pred_label = torch.load('pred_label_0.pt')



# generate the indices for sorting for each row
indices = torch.arange(pred_label.size(1)).repeat(pred_label.size(0), 1).to(device)

# for each row, combine the tensor and indices into a tuple
indexed_tensor = [list(zip(pred_label[i, :], indices[i, :])) for i in range(pred_label.size(0))]



# sort each row of indexed_tensor by the descending order of first element in each tuple
# and then by increasing order of the second element
for i in tqdm(range(len(indexed_tensor))):
    indexed_tensor[i] = sorted(indexed_tensor[i], key=lambda x: (-x[0], x[1]))



# save the indexed_tensor
torch.save(indexed_tensor, 'indexed_tensor_0.pt')
# # load the indexed_tensor
# indexed_tensor = torch.load('indexed_tensor_0.pt')



# get the indices of the sorted_tensor
sorted_tensor = [[indexed_tensor[i][j][1] for j in range(len(indexed_tensor[i]))] for i in range(len(indexed_tensor))]
# convert the elements in sorted_tensor to the int type
sorted_tensor = [list(map(int, sorted_tensor[i])) for i in range(len(sorted_tensor))]


# extract the [0,1,2] to be the first three elements in each row of sorted_tensor and then append the rest different elements in this row
first_list = [0,1,2]
for i in tqdm(range(len(sorted_tensor))):
    sorted_tensor[i] = first_list + [j for j in sorted_tensor[i] if j not in first_list]







# save the sorted_tensor
torch.save(sorted_tensor, 'sorted_tensor_0.pt')
# # load the sorted_tensor
# sorted_tensor = torch.load('sorted_tensor_0.pt')




# Now load the result of recall

# load the result of recall
with open('rule_recall_100_test.pkl', 'rb') as f:
    rule_recall_100_test = pickle.load(f)


# get the value of the corresponding indices in rule_recall_100_test['next_item_prediction']
sorted_recall_test = [[rule_recall_100_test['next_item_prediction'][i][j] for j in sorted_tensor[i]] for i in range(len(sorted_tensor))]


# replace the first elements in rule_recall_100_test['next_item_prediction'] with the sorted_recall_test
rule_recall_100_test['next_item_prediction'][:len(sorted_recall_test)] = sorted_recall_test





# save the rule_recall_100_test into a pickle file
with open('rule_recall_100_test_sorted.pkl', 'wb') as f:
    pickle.dump(rule_recall_100_test, f)








# load the rule_recall_100_test_sorted
with open('rule_recall_100_test_sorted.pkl', 'rb') as f:
    rule_recall_100_test_sorted = pickle.load(f)

rule_recall_100_test_sorted[['locale', 'next_item_prediction']].to_parquet('submission_task_test.parquet', engine='pyarrow')