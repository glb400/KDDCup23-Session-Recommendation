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
        "structure": "widen-avg-fixed emb-att-senet-residual 6-fc",
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
        prevs_id_num = input[:, :50]
        prevs_locale = input[:, 50:56]
        prevs_price = input[:, 56:77]
        next_id_num = input[:, 77:127]
        next_locale = input[:, 127:133]
        next_price = input[:, 133:154]
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
        self.attention = AveragePoolingLayerWithMask(input_dim, input_dim)
        self.se = SENet(input_dim)
        # use residual blocks
        self.residual_block1 = ResidualBlockWiden(2 * input_dim)
        self.residual_block2 = ResidualBlockWiden(2 * input_dim)
        self.residual_block3 = ResidualBlockWiden(2 * input_dim)
        self.residual_block4 = ResidualBlockWiden(2 * input_dim)
        self.residual_block5 = ResidualBlockWiden(2 * input_dim)
        self.residual_block6 = ResidualBlockWiden(2 * input_dim)
        # get the final 1-dimensional output
        self.fc = nn.Linear(2 * input_dim, output_dim)
        # use nn.Embedding to get the embedding of the input with 20 dimensions
        self.embed = nn.Embedding(vocab_size, 50)
        self.embed.weight.requires_grad = False
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
        # apply a series of residual blocks to the output of the SENet
        # the output is a 2D tensor with shape (batch_size, input_dim)
        output = self.residual_block1(output)
        output = self.residual_block2(output)
        output = self.residual_block3(output)
        output = self.residual_block4(output)
        output = self.residual_block5(output)
        output = self.residual_block6(output)
        # the output is a 2D tensor with shape (batch_size, 1)
        output = self.fc(output)
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


# define the training function
def train(model, data, mask, label, optimizer, criterion):
    # set the model to training mode
    model.train()
    dataset = MyDataset(data, mask, label)
    dataloader = DataLoader(dataset, batch_size=256, shuffle=True)
    for data_batch, mask_batch, label_batch in tqdm(dataloader):
        # move the data to the device
        data_batch, mask_batch, label_batch = data_batch.to(device), mask_batch.to(device), label_batch.to(device)
        # compute the output
        output = model(data_batch, mask_batch)
        # compute the loss
        loss = criterion(output, label_batch.squeeze().long())
        # compute the gradient
        optimizer.zero_grad()
        loss.backward()
        # update the parameters
        optimizer.step()
        # compute the accuracy
        pred_label = torch.argmax(output, dim=1)
        acc = accuracy_score(label_batch.cpu().numpy(), pred_label.cpu().numpy())
        wandb.log({'loss': loss.item(), 'acc': acc})
    return 




# START TRAINING

# set the random seed for reproducibility
np.random.seed(100)

# set the device
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')



# load all the training data
train_data = torch.load('final_aug_train_data.pt')
train_mask = torch.load('final_aug_train_mask.pt')
train_label = torch.load('final_aug_train_label.pt')

vocab_size=(int)(torch.max(train_data[:, :, 0])) + 1


# create the model
# input_dim = train_data.shape[2] + (50 - 1)
# because id is converted to id_embedding, so the input_dim should be 77
model = Model( input_dim=77, output_dim=2, vocab_size=vocab_size ) 
model.to(device)


# count the number of parameters of the model
num_params = sum(p.numel() for p in model.parameters() if p.requires_grad)

# count the number of trainable parameters of AttentionLayerWithMask layer
num_params_attention = sum(p.numel() for p in model.attention.parameters() if p.requires_grad)

# count the number of trainable parameters of SENet layer
num_params_se = sum(p.numel() for p in model.se.parameters() if p.requires_grad)




criterion = nn.CrossEntropyLoss()
optimizer = optim.Adam(model.parameters(), lr=0.001)

for epoch in range(10):
    print('epoch: {}'.format(epoch+1))
    # train the model using the training set
    train(model, train_data, train_mask, train_label, optimizer, criterion)
    # save the model
    torch.save(model.state_dict(), 'widen_avg_fixed_model_v1_6blks.pth')

# # load the model
# model.load_state_dict(torch.load('widen_avg_fixed_model_v1_6blks.pth'))
