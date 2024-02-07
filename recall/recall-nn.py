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
import random
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.metrics import accuracy_score


# define the target attention layer using Q, K, V
# the input is a 3D tensor with shape (batch_size, seq_len, input_dim)
# the output is a 2D tensor with shape (batch_size, output_dim)
class AttentionLayer(nn.Module):
    def __init__(self, input_dim, output_dim):
        super(AttentionLayer, self).__init__()
        self.input_dim = input_dim
        self.output_dim = output_dim
        self.Wq = nn.Linear(input_dim, output_dim)
        self.Wk = nn.Linear(input_dim, output_dim)
        self.Wv = nn.Linear(input_dim, output_dim)
        self.softmax = nn.Softmax(dim=1)
        self.tanh = nn.Tanh()
        self.dropout = nn.Dropout(p=0.5)
        
    def forward(self, input):
        # input is a 3D tensor with shape (batch_size, seq_len, input_dim)
        # Q, K, V are 3D tensors with shape (batch_size, seq_len, output_dim)
        Q = self.Wq(input[:, -1, :])
        K = self.Wk(input[:, :-1, :])
        V = self.Wv(input[:, :-1, :])
        # QK^T is a 3D tensor with shape (batch_size, seq_len, seq_len)
        # the softmax is applied to the last dimension
        # the output is a 3D tensor with shape (batch_size, seq_len, seq_len)
        QKt = torch.bmm(Q, K.transpose(1, 2))
        QKt = self.softmax(QKt)
        # QK^TV is a 3D tensor with shape (batch_size, seq_len, output_dim)
        # the output is a 3D tensor with shape (batch_size, seq_len, output_dim)
        QKtV = torch.bmm(QKt, V)
        # the output is a 2D tensor with shape (batch_size, output_dim)
        output = torch.sum(QKtV, dim=1)
        # the output is a 2D tensor with shape (batch_size, output_dim)
        output = self.tanh(output)
        # the output is a 2D tensor with shape (batch_size, output_dim)
        output = self.dropout(output)
        return output


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
        self.softmax = nn.Softmax()
        self.tanh = nn.Tanh()
        self.dropout = nn.Dropout(p=0.5)

    def forward(self, input, mask):
        # input is a 3D tensor with shape (batch_size, seq_len, input_dim)
        # mask is a 2D tensor with shape (batch_size, seq_len)
        # -------------------------------

        seq_len = torch.ones(mask.shape[0], dtype=torch.int32) * mask.shape[1]

        # use mask to mask out the padded values using 1e-9
        if mask is not None:
            # print(f'mask.shape: {mask.shape}')
            
            # get the number of 1 from mask
            # the output is a 1D tensor with shape (batch_size) in integer
            
            seq_len = torch.sum(mask, dim=1, dtype=torch.int32)

        output = torch.Tensor().to(device)

        print(f'YOYOYO input.shape: {input.shape}')

        for i in range(input.shape[0]):
            sqlen = seq_len[i]
            # print(f'sqlen: {sqlen}')

            q = self.Wq(input[i, sqlen-1, :].unsqueeze(0))
            k = self.Wk(input[i, :sqlen-1, :])
            v = self.Wv(input[i, :sqlen-1, :])
            # print(f'q.shape: {q.shape}')
            # print(f'k.shape: {k.shape}')

            qkt = torch.mm(q, k.transpose(0, 1))
            # print(f'qkt.shape: {qkt.shape}')

            # use softmax to normalize the weights
            qkt = self.softmax(qkt)
            # print(f'qkt.shape: {qkt.shape}')

            # QK^TV is a 3D tensor with shape (batch_size, seq_len, output_dim)
            # the output is a 3D tensor with shape (batch_size, seq_len, output_dim)
            qktv = torch.mm(qkt, v)
            # print(f'qktv.shape: {qktv.shape}')

            output = torch.cat((output, qktv), dim=0)
            # print(f'output.shape: {output.shape}')

        
        
        
        
        # This is for self-attention version:

        # # Q, K, V are 3D tensors with shape (batch_size, seq_len, output_dim)
        # Q = self.Wq(input[:, -1, :].unsqueeze(1))
        # K = self.Wk(input[:, :-1, :])
        # V = self.Wv(input[:, :-1, :])
        # print(f'Q.shape: {Q.shape}')
        # print(f'K.shape: {K.shape}')

        # # QK^T is a 3D tensor with shape (batch_size, seq_len, seq_len)
        # # the softmax is applied to the last dimension
        # # the output is a 3D tensor with shape (batch_size, seq_len, seq_len)
        # QKt = torch.bmm(Q, K.transpose(1, 2))

        # # use mask to mask out the padded values using 1e-9
        # if mask is not None:
        #     print(f'mask.shape: {mask.shape}')
        #     mask = mask.unsqueeze(1)
        #     print(f'QKt.shape: {QKt.shape}')
        #     print(f'mask.shape: {mask.shape}')
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







# # define the SENet
# # the input is a 2D tensor with shape (batch_size, input_dim)
# # the output is a 2D tensor with shape (batch_size, output_dim) with output_dim = input_dim
# class SENet(nn.Module):
#     def __init__(self, input_dim):
#         super(SENet, self).__init__()
#         self.input_dim = input_dim
#         self.fc1 = nn.Linear(input_dim, input_dim)
#         self.fc2 = nn.Linear(input_dim, input_dim)
#         self.r = 16

#     def forward(self, input):
#         # input is a 2D tensor with shape (batch_size, input_dim)
#         # use average/max/min pooling to get three 2D tensors with shape (batch_size, 1)
#         avg_pool = torch.mean(input, dim=1, keepdim=True)
#         print(f'avg_pool.shape: {avg_pool.shape}')
        
#         max_pool, _ = torch.max(input, dim=1, keepdim=True)
#         print(f'max_pool.shape: {max_pool.shape}')

#         min_pool, _ = torch.min(input, dim=1, keepdim=True)
#         print(f'min_pool.shape: {min_pool.shape}')

#         # concatenate the three 2D tensors into a 2D tensor with shape (batch_size, 3)
#         se_weights = torch.cat((avg_pool, max_pool, min_pool), dim=1)
#         print(f'se_weights.shape: {se_weights.shape}')

#         # use a fully connected layer with relu activation to shrink the 2D tensor into a 2D tensor with shape (batch_size, input_dim // 16)
#         se_weights = self.fc1(se_weights)
#         se_weights = F.relu(se_weights)
#         # use a fully connected layer with sigmoid activation to expand the 2D tensor into a 2D tensor with shape (batch_size, input_dim)
#         se_weights = self.fc2(se_weights)
#         se_weights = nn.Sigmoid(se_weights)
#         # use row-wise mutiplication to get a 2D tensor with shape (batch_size, input_dim)
#         # row-wise multiply se_weights with input
#         output = torch.mul(se_weights, input)
#         # the output is a 2D tensor with shape (batch_size, input_dim)
#         output = nn.dropout(output, p=0.5)
#         return output







# define the model
# the input is a 3D tensor with shape (batch_size, seq_len, input_dim)
# the output is a tensor with shape (batch_size, 2, 200)
# which represents two (batch_size, 200) vectors for the prev_items(session) and the next_item
class Model(nn.Module):
    def __init__(self, input_dim, output_dim):
        super(Model, self).__init__()
        self.input_dim = input_dim
        self.output_dim = output_dim
        # self.attention = AttentionLayer(input_dim, input_dim)
        
        self.attention = AttentionLayerWithMask(input_dim, input_dim)
        
        # self.se = SENet(input_dim)

        # reduce the dimension of the output of SENet to 1/2 of the input dimension
        self.fc1 = nn.Linear(input_dim, input_dim // 2)
        self.relu1 = nn.ReLU()
        # reduce the dimension of the output of SENet to 1/4 of the input dimension
        self.fc2 = nn.Linear(input_dim // 2, input_dim // 4)
        self.relu2 = nn.ReLU()
        # reduce the dimension of the output of SENet to 1/8 of the input dimension
        self.fc3 = nn.Linear(input_dim // 4, 200)
        self.relu3 = nn.ReLU()
        
    def forward(self, input, mask):
        # input is a 3D tensor with shape (batch_size, seq_len, input_dim)
        # the output is a 2D tensor with shape (batch_size, 1)
        # the output is a probability between 0 and 1
        # ----------------------------------------------
        # apply the attention layer to the input
        # the output is a 2D tensor with shape (batch_size, input_dim)
        next_item_output = input[:, -1, :].squeeze()

        prev_items_output = self.attention(input, mask)
        


        # # apply the SENet to the output of the attention layer
        # # the output is a 2D tensor with shape (batch_size, input_dim)
        # next_item_output = self.se(next_item_output)
        # prev_items_output = self.se(prev_items_output)



        # apply a fully connected layer with relu activation to the output of the SENet
        # the output is a 2D tensor with shape (batch_size, input_dim // 2)
        next_item_output = self.fc1(next_item_output)
        next_item_output = self.relu1(next_item_output)

        prev_items_output = self.fc1(prev_items_output)
        prev_items_output = self.relu1(prev_items_output)


        # apply a fully connected layer with relu activation to the output of the SENet
        # the output is a 2D tensor with shape (batch_size, input_dim // 4)
        next_item_output = self.fc2(next_item_output)
        next_item_output = self.relu2(next_item_output)

        prev_items_output = self.fc2(prev_items_output)
        prev_items_output = self.relu2(prev_items_output)


        # apply a fully connected layer with relu activation to the output of the SENet
        # the output is a 2D tensor with shape (batch_size, input_dim // 8)
        next_item_output = self.fc3(next_item_output)
        next_item_output = self.relu3(next_item_output)

        prev_items_output = self.fc3(prev_items_output)
        prev_items_output = self.relu3(prev_items_output)



        # concatenate the two 2D tensors into a 3D tensor with shape (batch_size, 2, (input_dim // 8))
        output = torch.cat((prev_items_output, next_item_output), dim=1)

        # reshape output to a 3D tensor with shape (batch_size, 2, 200)
        output = output.reshape(output.shape[0], 2, 200)

        # This is for classification task
        # # use softmax to convert the output to a probability
        # # the output is a 2D tensor with shape (batch_size, 1)
        # output = F.softmax(output, dim=1)


        return output


# define the training function
def train(model, data, mask, optimizer, criterion, device):
    # set the model to training mode
    model.train()
    # train the model using the entire training data for one epoch

    # generate label for the training data
    label = torch.tensor([]).to(device)

    # generate mask for the training data
    augmented_mask = torch.tensor([]).to(device)

    # train in batches of 256 examples
    batch_size = 256

    for i in range(0, data.shape[0], batch_size):
        # compute the starting index of the batch
        start = i
        end = min(data.shape[0], start + batch_size)

        # extract a batch of training data and mask
        data_batch = data[start:end, :, :].to(device)
        mask_batch = mask[start:end, :].to(device)

        # generate the label for the batch
        label_batch = torch.tensor([]).to(device)


        # randomly construct negative samples
        for item in data_batch:
            # append the positive label for the item
            label_batch = torch.cat((label_batch, torch.tensor([1]).to(device)), dim=0)






            # # randomly choose one different items from the data_batch
            # random_item = data_batch[random.randint(0, data_batch.shape[0] - 1)].squeeze()
            # while torch.equal(item, random_item):
            #     random_item = data_batch[random.randint(0, data_batch.shape[0] - 1)].squeeze() 

            # # crossover the two items to get two negative sample
            # if not torch.equal(item[-1, :], random_item[-1, :]):
            #     neg_item1 = torch.cat((item[:-1, :], random_item[-1, :].unsqueeze(0)), dim=0)
            #     print(f'neg_item1: {neg_item1}')

            #     data_batch = torch.cat((data_batch, neg_item1.unsqueeze(0)), dim=0)

            #     label_batch = torch.cat((label_batch, torch.tensor([0]).to(device)), dim=0)

            #     # generate mask_batch for item neg_item1

            #     neg_mask = torch.zeros(1, train_data.shape[1]).to(device)


            #     for i in range(train_data.shape[1]):
            #         if not torch.equal(neg_item1[i, :], torch.zeros(neg_item1.shape[1], device=device)):
            #             neg_mask[:, i] = 1

            #     mask_batch = torch.cat((mask_batch, neg_mask), dim=0)

            #     # neg_item2 = torch.cat((random_item[:-1, :].unsqueeze(0), item[-1, :]), dim=0)
            #     # data_batch = torch.cat((data_batch, neg_item2.unsqueeze(0)), dim=0)
            #     # label_batch = torch.cat((label_batch, torch.tensor([0]).to(device)), dim=0)




        # compute the output
        output = model(data_batch, mask_batch)
        print(f'output shape: {output.shape}')

        # compute the loss
        loss = criterion(output, label_batch)

        # compute the gradient
        optimizer.zero_grad()
        loss.backward()

        # update the parameters
        optimizer.step()



        # # compute the accuracy
        # pred_label = torch.argmax(output, dim=1).to(device)

        # print(f'label_batch shape: {label_batch.shape}')

        # print(f'pred_label shape: {pred_label.shape}')

        # acc = accuracy_score(label_batch.cpu().numpy, pred_label.cpu().numpy())
 



        # add label_batch to label
        label = torch.cat((label, label_batch), dim=0)

        # add mask_batch to mask
        augmented_mask = torch.cat((augmented_mask, mask_batch), dim=0)

        # # print the loss and accuracy
        # print('loss: {:.4f}, acc: {:.4f}'.format(loss.item(), acc))

    # then print the average loss and accuracy for the entire training set
    # compute the output
    output = model(data, augmented_mask)

    # compute the loss
    loss = criterion(output, label)



    # # compute the accuracy
    # pred_label = torch.argmax(output, dim=1).to(device)

    # acc = accuracy_score(label_batch.cpu().numpy, pred_label.cpu().numpy())



    return loss.item()


# begin to train the model

# set the random seed for reproducibility
np.random.seed(100)

# set the device
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')


# train_data = pd.read_pickle('df_sess_tot_feature0.pkl')

# # save the first 10000 of train_data to train_data.pkl
# train_data = train_data[:10000]
# train_data.to_pickle('shabi.pkl')



train_data = pd.read_pickle('df_sess_tot_feature0.pkl')

train_data = train_data[:1000]

# # load the train_data
# train_data = pd.DataFrame()

# # read all the train_data from df_sess_tot_feature0 to df_sess_tot_feature20
# for i in range(20):
#     # append the train_data to train_data
#     train_data = train_data.append(pd.read_pickle('df_sess_tot_feature{}'.format(i)+'.pkl'))


train_data = train_data.apply(lambda x: np.array(x)).apply(lambda x: torch.from_numpy(x))

# generate padding for the train_data on the 2th dimension in (batch_size, seq_len, embedding_dim)
# get the max seq_len
maxlen = train_data.apply(lambda x: x.shape[0]).max()
print(f'maxlen: {maxlen}')

# pad each item of train_data using torch.nn.functional.pad on the 1th dimension with value 0
train_data = train_data.apply(lambda x: F.pad(x, (0, 0, 0, maxlen - x.shape[0]), 'constant', 0))

# convert the train_data to torch.tensor
train_data = torch.stack(tuple(train_data.values))

# move the train_data to the device and convert the dtype to torch.float32
train_data = train_data.to(device).float()

# construct mask for train_data
train_mask = torch.zeros(train_data.shape[0], train_data.shape[1])
for i in range(train_data.shape[0]):
    for j in range(train_data.shape[1]):
        if not torch.equal(train_data[i, j, :], torch.zeros(train_data.shape[2], device=device)):
            train_mask[i, j] = 1


# create the model
model = Model(input_dim=1544, output_dim=200)

# move the model to the device
model.to(device)

# # define the loss function
# criterion = nn.CrossEntropyLoss()

# define a new criterion that computes the inner product of the outputs, i.e., x1 and x2
def recall_criterion(output, label):
    # compute the consine similarity of the outputs
    consine_similarity = torch.sum(output[:, 0, :] * output[:, 1, :], dim=1) / ((torch.norm(output[:, 0, :], dim=1) * torch.norm(output[:, 1, :], dim=1)))
    
    print(f'consine_similarity shape: {consine_similarity.shape}')
    
    print(f'label shape: {label.shape}')

    # compute the categorical cross entropy loss
    loss = -torch.sum(label * torch.log(consine_similarity) + (1 - label) * torch.log(1 - consine_similarity))

    return loss


# define the optimizer
optimizer = optim.Adam(model.parameters(), lr=0.001)

# train the model for 10 epochs
for epoch in range(10):
    print('epoch: {}'.format(epoch+1))
    
    # train the model using the training set
    
    loss = train(model, train_data, train_mask, optimizer, recall_criterion, device)
    
    # print('loss: {:.4f}, acc: {:.4f}'.format(loss, acc))
    print('loss: {:.4f}'.format(loss))


# save the model
torch.save(model.state_dict(), 'model.pth')

# load the model
model.load_state_dict(torch.load('model.pth'))



# load the test data
# test_data = pd.read_pickle('df_sess_tot_feature0.pkl')

# compute the output for test data
# output = model(test_data, test_mask)

# get the predicted embedding















