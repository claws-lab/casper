#@contact    Sejoon Oh (soh337@gatech.edu), Georgia Institute of Technology
#@version    1.0
#@date       2022-08-04
#Rank List Sensitivity of Recommender Systems to Interaction Perturbations
#This software is free of charge under research purposes.
#For commercial purposes, please contact the main author.

import torch
import torch.nn as nn
from torch.nn import functional as F
from torch import optim
import numpy as np
import math, random
from collections import defaultdict
import rbo
import time

class LSTM(nn.Module):
    def __init__(self, data, input_size, output_size, hidden_dim, args, n_layers=1, device="cpu"):
        super(LSTM, self).__init__()
        
        self.num_items = output_size
        self.device = device 
        self.emb_length = input_size
        self.batch_size = args.batch_size
        self.item_emb = nn.Embedding(self.num_items, self.emb_length,padding_idx=0)
        self.hidden_dim = hidden_dim
        self.n_layers = n_layers
        self.learning_rate = args.learning_rate

        self.LSTM = nn.LSTM(input_size, hidden_dim, n_layers, batch_first=True)
        self.fc = nn.Linear(hidden_dim, output_size)

    def forward(self, x):
        batch_size = x.size(0)
        hidden = self.init_hidden(batch_size)
        out, hidden = self.LSTM(x, hidden)
        inp = out[:, -1, :].contiguous().view(-1, self.hidden_dim)
        out = self.fc(inp)
        return out, hidden

    def init_hidden(self, batch_size):
        hidden = (torch.zeros(self.n_layers, batch_size, self.hidden_dim).to(self.device).detach(), torch.zeros(self.n_layers, batch_size, self.hidden_dim).to(self.device).detach())
        return hidden

    def traintest(self, train,test,epochs, original_probs):
        
        total_train_num = len(train)
        current_labels = []
        for i in range(total_train_num):
            train[i][0] = self.item_emb(torch.LongTensor(train[i][0]).to(self.device))
            current_labels.append(train[i][1])
        train_out = torch.LongTensor(current_labels).to(self.device)
        
        total_test_num = len(test)
        current_labels = []
        for i in range(total_test_num):
            test[i][0] = self.item_emb(torch.LongTensor(test[i][0]).to(self.device))
            current_labels.append(test[i][1])
        test_out = torch.LongTensor(current_labels).to(self.device)

        print("number of training&test data={},{}".format(total_train_num,total_test_num))

        criterion = nn.CrossEntropyLoss()
        optimizer = optim.Adam(self.parameters(), lr=self.learning_rate, weight_decay=1e-5)
        start_time = time.time()

        probs = [0 for i in range(total_test_num)]
        MRR,HITS = 0,0
        for epoch in range(epochs):
            train_loss=0
            for iteration in range(int(total_train_num/self.batch_size)+1):
                st_idx,ed_idx = iteration*self.batch_size, (iteration+1)*self.batch_size
                if ed_idx>total_train_num:
                    ed_idx = total_train_num
                        
                optimizer.zero_grad()
                output, hidden = self.forward(torch.stack([train[i][0] for i in range(st_idx,ed_idx)],dim=0).detach())
                loss = criterion(output, train_out[st_idx:ed_idx])
                loss.backward()  
                train_loss += loss.item()
                optimizer.step()
                 
            if epoch % 10 == 0:
                print("Epoch {}\tTrain Loss: {}\tElapsed time: {}".format(epoch, train_loss/total_train_num, time.time() - start_time))
                start_time = time.time()
        
        for iteration in range(int(total_test_num / self.batch_size) + 1):
            st_idx, ed_idx = iteration * self.batch_size, (iteration + 1) * self.batch_size
            if ed_idx > total_test_num:
                ed_idx = total_test_num
            output, hidden = self.forward(torch.stack([test[i][0] for i in range(st_idx,ed_idx)],dim=0).detach())
            test_loss = criterion(output, test_out[st_idx:ed_idx])

            output = output.view(-1, self.num_items)
            prob = nn.functional.softmax(output, dim=1).data.cpu()
            np_prob = prob.numpy()
            current_val = np.zeros((np_prob.shape[0],1))
            for i in range(st_idx,ed_idx):
                current_test_label = test[i][1]
                current_val[i-st_idx,0] = np_prob[i-st_idx,current_test_label]
            
            new_prob = np_prob - current_val
            ranks = np.count_nonzero(new_prob>0,axis=1)
            
            for i in range(st_idx,ed_idx):
                rank = ranks[i-st_idx]+1 
                MRR += 1/rank
                HITS += (1 if rank<=10 else 0)
                probs[i] = np_prob[i-st_idx,:]

        MRR /= total_test_num
        HITS /= total_test_num
        print('Test MRR = {}\tTest Recall@10 = {}\n'.format(MRR,HITS))

        rbos,jaccards = [],[]
        avg_rbo,avg_jaccard = -1,-1
        if original_probs!=-1:
            rank1,rank2 = np.argsort(-np.array(original_probs),axis=1),np.argsort(-np.array(probs),axis=1)
            for i in range(total_test_num):
                ground_truth = test[i][1]
                RBO = rbo.RankingSimilarity(rank1[i,:], rank2[i,:]).rbo()
                jaccard = np.intersect1d(rank1[i,:10],rank2[i,:10]).shape[0]/np.union1d(rank1[i,:10],rank2[i,:10]).shape[0]
                rbos.append(RBO)
                jaccards.append(jaccard)
            avg_rbo,avg_jaccard = np.average(rbos),np.average(jaccards)
            print('RLS metrics: (RBO,Jaccard@10) = {},{}\n'.format(avg_rbo,avg_jaccard))

        return [probs,[avg_rbo,avg_jaccard],[MRR,HITS]]

