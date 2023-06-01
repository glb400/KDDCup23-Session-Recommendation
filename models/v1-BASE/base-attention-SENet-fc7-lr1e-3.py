# build a classification model with inputs that have 1538 dimensions using pytorch
# the model first needs to use self-attention mechanism to pool the input vectors into one vector with same dimension
# then the pooled vector is fed into SENet
# the output of SENet is then fed into a fully connected layer to get the final output
# the model is trained with cross entropy loss
# the model is evaluated with accuracy
# the model is saved to a file
# the model is loaded from a file

import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader
import os
import wandb

import random
from tqdm import *
import pickle
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.metrics import accuracy_score


run = wandb.init(
    # Set the project where this run will be logged
    project="kddcup-v1",
    # Track hyperparameters and run metadata
    config={
        "learning_rate": 0.001,
        "epochs": 10,
        "batch_size": 256,
        "structure": "att-senet-fc-7",
    })

# # define the original target attention layer using Q, K, V
# # the input is a 3D tensor with shape (batch_size, seq_len, input_dim)
# # the output is a 2D tensor with shape (batch_size, output_dim)
# class AttentionLayer(nn.Module):
#     def __init__(self, input_dim, output_dim):
#         super(AttentionLayer, self).__init__()
#         self.input_dim = input_dim
#         self.output_dim = output_dim
#         self.Wq = nn.Linear(input_dim, output_dim)
#         self.Wk = nn.Linear(input_dim, output_dim)
#         self.Wv = nn.Linear(input_dim, output_dim)
#         self.softmax = nn.Softmax(dim=1)
#         self.tanh = nn.Tanh()
#         self.dropout = nn.Dropout(p=0.5)
        
#     def forward(self, input):
#         # input is a 3D tensor with shape (batch_size, seq_len, input_dim)
#         # Q, K, V are 3D tensors with shape (batch_size, seq_len, output_dim)
#         Q = self.Wq(input[:, -1, :])
#         K = self.Wk(input[:, :-1, :])
#         V = self.Wv(input[:, :-1, :])

#         # QK^T is a 3D tensor with shape (batch_size, seq_len, seq_len)
#         # the softmax is applied to the last dimension
#         # the output is a 3D tensor with shape (batch_size, seq_len, seq_len)
#         QKt = torch.bmm(Q, K.transpose(1, 2))
#         QKt = self.softmax(QKt)
        
#         # QK^TV is a 3D tensor with shape (batch_size, seq_len, output_dim)
#         # the output is a 3D tensor with shape (batch_size, seq_len, output_dim)
#         QKtV = torch.bmm(QKt, V)
        
#         # the output is a 2D tensor with shape (batch_size, output_dim)
#         output = torch.sum(QKtV, dim=1)
        
#         # the output is a 2D tensor with shape (batch_size, output_dim)
#         output = self.tanh(output)
        
#         # the output is a 2D tensor with shape (batch_size, output_dim)
#         output = self.dropout(output)
        
#         return output



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
            # print(f'sqlen: {sqlen}')
            # print(f'input.shape: {input.shape}')
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

        # # This is for self-attention version
        # # Q, K, V are 3D tensors with shape (batch_size, seq_len, output_dim)
        # Q = self.Wq(input[:, -1, :].unsqueeze(1))
        # K = self.Wk(input[:, :-1, :])
        # V = self.Wv(input[:, :-1, :])
        
        # # QK^T is a 3D tensor with shape (batch_size, seq_len, seq_len)
        # # the softmax is applied to the last dimension
        # # the output is a 3D tensor with shape (batch_size, seq_len, seq_len)
        # QKt = torch.bmm(Q, K.transpose(1, 2))
        
        # # use mask to mask out the padded values using 1e-9
        # if mask is not None:
        #     print(f'mask.shape: {mask.shape}')
        #     mask = mask.unsqueeze(1)
        #     QKt = QKt.masked_fill(mask == 0, 1e-9)
        
        # # use softmax to normalize the weights
        # QKt = self.softmax(QKt, dim=-1)
        
        # # QK^TV is a 3D tensor with shape (batch_size, seq_len, output_dim)
        # # the output is a 3D tensor with shape (batch_size, seq_len, output_dim)
        # QKtV = torch.bmm(QKt, V)
        
        # # the output is a 2D tensor with shape (batch_size, output_dim)
        # output = torch.sum(QKtV, dim=1)
        
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
        self.fc1_2 = nn.Linear(input_dim, input_dim)
        # reduce the dimension of the output of SENet to 1/4 of the input dimension
        self.fc2 = nn.Linear(input_dim, input_dim // 2)
        self.fc2_2 = nn.Linear(input_dim // 2, input_dim // 2)
        # reduce the dimension of the output of SENet to 1/8 of the input dimension
        self.fc3 = nn.Linear(input_dim // 2, input_dim // 4)
        self.fc3_2 = nn.Linear(input_dim // 4, input_dim // 4)
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

        output = self.fc1_2(output)
        output = F.relu(output)

        # the output is a 2D tensor with shape (batch_size, input_dim // 4)
        output = self.fc2(output)
        output = F.relu(output)

        output = self.fc2_2(output)
        output = F.relu(output)
        
        # the output is a 2D tensor with shape (batch_size, input_dim // 8)
        output = self.fc3(output)
        output = F.relu(output)
        
        output = self.fc3_2(output)
        output = F.relu(output)
        # the output is a 2D tensor with shape (batch_size, 1)
        output = self.fc4(output)

        # This is for classification task
        # use softmax to convert the output to a probability
        # the output is a 2D tensor with shape (batch_size, 2)
        output = F.softmax(output, dim=1)
        
        return output




# # define the function which augment the training data on device
# def augment_data(train_data, train_mask):
#     # train_data, train_mask = train_data.to(device), train_mask.to(device)
#     aug_data, aug_label, aug_mask = torch.tensor([]).to(device), torch.tensor([]).to(device), torch.tensor([]).to(device)

#     # count the number of saved samples
#     count = 0

#     for i in tqdm(range(train_data.shape[0])):
#         # append positive samples
#         item, mask, label = train_data[i, :, :].to(device), train_mask[i, :].to(device), torch.tensor([1]).to(device)
#         aug_data = torch.cat((aug_data, item.unsqueeze(0).to(device)), dim=0)
#         aug_mask = torch.cat((aug_mask, mask.unsqueeze(0).to(device)), dim=0)
#         aug_label = torch.cat((aug_label, label.unsqueeze(0).to(device)), dim=0).to(device)

#     	# generate negative samples
#         random_item = train_data[random.randint(0, train_data.shape[0] - 1)].squeeze().to(device)
#         while torch.equal(item, random_item):
#             random_item = train_data[random.randint(0, train_data.shape[0] - 1)].squeeze().to(device)

#         if not torch.equal(item[-1, :], random_item[-1, :]):
#             neg_item, neg_mask, neg_label = torch.cat((item[:-1, :], random_item[-1, :].unsqueeze(0)), dim=0), train_mask[i, :], torch.tensor([0])
#             aug_data = torch.cat((aug_data, neg_item.unsqueeze(0).to(device)), dim=0)
#             aug_mask = torch.cat((aug_mask, neg_mask.unsqueeze(0).to(device)), dim=0)
#             aug_label = torch.cat((aug_label, neg_label.unsqueeze(0).to(device)), dim=0)

#         # save the augmented data every 100,000 iterations
#         if i != 0 and i % 5e4 == 0:
#             torch.save(aug_data, 'aug_data_' + str(count) + '.pt')
#             torch.save(aug_mask, 'aug_mask_' + str(count) + '.pt')
#             torch.save(aug_label, 'aug_label_' + str(count) + '.pt')
#             aug_data, aug_label, aug_mask = torch.tensor([]).to(device), torch.tensor([]).to(device), torch.tensor([]).to(device)
#             count += 1

#     torch.save(aug_data, 'aug_data_' + str(count) + '.pt')
#     torch.save(aug_mask, 'aug_mask_' + str(count) + '.pt')
#     torch.save(aug_label, 'aug_label_' + str(count) + '.pt')
#     return aug_data, aug_mask, aug_label






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
        
        # print(f'output shape: {output.shape}')
        # print(f'label_batch shape: {label_batch.shape}')

        
        # print(f'output[0]: {output[0]}')
        # print(f'label_batch[0]: {label_batch[0]}')
        
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

        # print the loss and accuracy
        # print('loss: {:.4f}, acc: {:.4f}'.format(loss.item(), acc))

        wandb.log({'loss': loss.item(), 'acc': acc})

    # # print the average loss and accuracy for the entire training set
    # # randomly choose 100000 samples from the training set
    # random_indices = np.random.choice(data.shape[0], 100000, replace=False)

    # valid_data, valid_mask, valid_label = data[random_indices, :], mask[random_indices, :], label[random_indices]

    # # compute the output
    # output = model(valid_data.to(device), valid_mask.to(device))

    # # compute the loss
    # loss = criterion(output, valid_label.squeeze().long().to(device))

    # # compute the accuracy
    # pred_label = torch.argmax(output, dim=1)

    # acc = accuracy_score(valid_label.cpu().numpy(), pred_label.cpu().numpy())

    return 




# START TRAINING

# set the random seed for reproducibility
np.random.seed(100)

# set the device
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')


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
# print(f'maxlen: {maxlen}')
wandb.log({'maxlen': maxlen})



# # # delete the train_data with shape[0] larger than maxlen
# # train_data = train_data[train_data.apply(lambda x: x.shape[0]) <= maxlen]

# # # pad each item of train_data using torch.nn.functional.pad on the 1th dimension with value 0
# # train_data = train_data.apply(lambda x: F.pad(x, (0, 0, 0, maxlen - x.shape[0]), 'constant', 0))

# # # convert the train_data to torch.tensor
# # train_data = torch.stack(tuple(train_data.values))

# # # save the train_data
# # torch.save(train_data, 'train_data.pt')

# # load the train_data
# train_data = torch.load('train_data.pt')



# # # construct mask for train_data
# # train_mask = torch.zeros(train_data.shape[0], train_data.shape[1])

# # for i in tqdm(range(train_data.shape[0])):
# #     for j in range(train_data.shape[1]):
# #         if not torch.equal(train_data[i, j, :], torch.zeros(train_data.shape[2], device=device)):
# #             train_mask[i, j] = 1

# # # save the train_mask
# # torch.save(train_mask, 'train_mask.pt')
# # print(f'train_mask shape: {train_mask.shape}')

# # load the train_mask
# train_mask = torch.load('train_mask.pt')

# # augment train data and mask
# train_data, train_mask, train_label = augment_shuffled_data(train_data, train_mask)





# load all the training data
train_data = torch.load('final_aug_train_data.pt')
train_mask = torch.load('final_aug_train_mask.pt')
train_label = torch.load('final_aug_train_label.pt')

vocab_size=(int)(torch.max(train_data[:, :, 0])) + 1



# train_data = train_data[:1000]
# train_mask = train_mask[:1000]
# train_label = train_label[:1000]

# vocab_size = 1000



# create the model
# input_dim = train_data.shape[2] + (20 - 1)
# because id is converted to id_embedding, so the input_dim should be 47
model = Model( input_dim=47, output_dim=2, vocab_size=vocab_size ) 
model.to(device)

criterion = nn.CrossEntropyLoss()

optimizer = optim.Adam(model.parameters(), lr=0.001)

for epoch in range(5):
    print('epoch: {}'.format(epoch+1))
    
    # train the model using the training set
    train(model, train_data, train_mask, train_label, optimizer, criterion)
    


# save the model
torch.save(model.state_dict(), 'model_v2.pth')

# # load the model
# model.load_state_dict(torch.load('model_v2.pth'))
