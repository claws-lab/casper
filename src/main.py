#@contact    Sejoon Oh (soh337@gatech.edu), Georgia Institute of Technology
#@version    1.0
#@date       2022-08-04
#Rank List Sensitivity of Recommender Systems to Interaction Perturbations
#This software is free of charge under research purposes.
#For commercial purposes, please contact the main author.

import torch
import torch.nn as nn
import numpy as np
import pandas as pd
import copy
import argparse
import os
from util import *
from lstm import *

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--data_path',type=str,default='data/wikipedia.tsv',help='path of the dataset (it should be tsv format; see data/wikipedia.tsv')
    parser.add_argument('--num_perturbation', default=1, type = int, help='Number of input perturbations interactions')
    parser.add_argument('--gpu',default='0',type=str,help='GPU number that will be used')
    parser.add_argument('--output',type=str, help = "path of the model stability result")
    parser.add_argument('--epochs', default=50, type = int, help='number of training epochs')
    parser.add_argument('--max_seq_len', default=50, type = int, help='maximum sequence length for CASPER perturbation')
    parser.add_argument('--test_data_ratio', default=0.1, type = float, help='last K% of interactions of each user will be used as test data')
    parser.add_argument('--batch_size', default=1024, type = int, help='mini-batch size for training')
    parser.add_argument('--learning_rate', default=0.001, type = float, help='learning rate for training')

    args = parser.parse_args()
    os.environ["CUDA_DEVICE_ORDER"]="PCI_BUS_ID"
    os.environ["CUDA_VISIBLE_DEVICES"]=args.gpu

    output_path = args.output
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(device)
    
    #Reading & cleaning the dataset
    raw_data = pd.read_csv(args.data_path, sep='\t', header=None).values[:,1:]
    look_back = args.max_seq_len
    processed_data = data_cleaning(raw_data, args.test_data_ratio)

    f = open(output_path,'w')

    #Define LSTM model and train it with the original training data
    (train,test) = train_test_generator(processed_data,look_back)
    model = LSTM(data = processed_data,input_size=128, output_size=len(np.unique(processed_data[:,1]))+1, hidden_dim=64, n_layers=1, device=device,args=args).to(device)
    model.LSTM.flatten_parameters()
    print(model)
    original_model = copy.deepcopy(model)

    #traintest function performs model training and return only next-item metrics (RLS cannot be obtained here)
    [original_probs,temp,[MRR,Recall]] = model.traintest(train=train,test=test,epochs = args.epochs, original_probs=-1)
    print('[Without perturbation] Avg MRR = {}\tAvg Recall@10 = {}\n'.format(MRR,Recall),file=f,flush=True)
    print('Original model training is done. Finding perturbations now.\n')
    
    #Creating IDAG
    user_seq,item_seq = defaultdict(list),defaultdict(list)
    in_degree, num_child = np.zeros(processed_data.shape[0]),np.zeros(processed_data.shape[0])
    edges = defaultdict(list)
    count = 0
    for i in range(processed_data.shape[0]):
        in_degree[i]=-1
        if processed_data[i,3]==0:
            count += 1
            user,item = int(processed_data[i,0]),int(processed_data[i,1])
            user_seq[user].append(i)
            item_seq[item].append(i)
            in_degree[i] = 0
    
    for user in user_seq.keys():
        cur_list = user_seq[user]
        st_idx = len(cur_list)-look_back+1
        if st_idx<0:
            st_idx = 0
        for i in range(st_idx,len(cur_list)-1):
            j,k = cur_list[i],cur_list[i+1]
            in_degree[k] += 1
            edges[j].append(k)

    for item in item_seq.keys():
        cur_list = item_seq[item]
        st_idx = len(cur_list)-look_back+1
        if st_idx<0:
            st_idx = 0
        for i in range(st_idx,len(cur_list)-1):
            j,k = cur_list[i],cur_list[i+1]
            in_degree[k] += 1
            edges[j].append(k)
    
    queue = []
    for i in range(processed_data.shape[0]):
        if in_degree[i]==0 and processed_data[i,3]==0:
            queue.append(i)

    #Finding an interaction that has the largest number of descendants
    while len(queue)!=0:
        root = queue.pop(0)
        check = np.zeros(processed_data.shape[0])
        check[root]=1
        q2 = [root]
        num_desc = 1
        #performing BFS from a current node(root)
        while len(q2)!=0:
            now = q2.pop(0)
            for node in edges[now]:
                if check[node]==0:
                    check[node]=1
                    q2.append(node)
                    num_desc += 1
        num_child[root] = num_desc

    #Interactions found by CASPER perturbation
    chosen = np.argsort(num_child)[-args.num_perturbation:]
    
    del_indices = []
    for idx in chosen:
        maxv,maxp = num_child[idx],idx
        user,item,time = int(processed_data[maxp,0]),int(processed_data[maxp,1]),processed_data[maxp,2]
        del_indices.append(maxp)
        print('[CASPER,Deletion] interaction {}=({},{},{}) is deleted\n'.format(maxp,user,item,time),file=f,flush=True)

    #Delete interactions found by CASPER
    new_data = np.delete(processed_data,del_indices,0)

    print('Perturbations are done. Retraining the model and calculate RLS and next-item metrics.\n')
 
    (train,test) = train_test_generator(new_data,look_back)
    model = copy.deepcopy(original_model)
    model.LSTM.flatten_parameters()
    [current_probs,[RBO,Jaccard],[MRR,Recall]] = model.traintest(train=train,test=test,epochs = args.epochs, original_probs=original_probs)
    print('[With CASPER perturbation] Avg MRR = {}\tAvg Recall@10 = {}\n'.format(MRR,Recall),file=f,flush=True)
    print('\nRanklist Sensitivity Metrics (RBO, Jaccard@10) = {},{}\n'.format(RBO,Jaccard),file = f,flush=True)       

if __name__ == "__main__":
    main()
