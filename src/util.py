#@contact    Sejoon Oh (soh337@gatech.edu), Georgia Institute of Technology
#@version    1.0
#@date       2022-08-04
#Rank List Sensitivity of Recommender Systems to Interaction Perturbations
#This software is free of charge under research purposes.
#For commercial purposes, please contact the main author.
import numpy as np

def data_cleaning(raw_data,test_data_ratio):
    unique_users = sorted(list(set(raw_data[:, 0])))
    unique_items = sorted(list(set(raw_data[:, 1])))
    user_dic = {user:idx for (idx,user) in enumerate(unique_users)}
    item_dic = {item:idx for (idx,item) in enumerate(unique_items)}
    for (idx, row) in enumerate(raw_data):
        user,item,time = user_dic[row[0]],item_dic[row[1]],row[2]
        raw_data[idx,0],raw_data[idx,1] = int(user),int(item)
 
    (users,counts) = np.unique(raw_data[:,0],return_counts = True)
    
    users = users[counts>=10]

    sequence_dic =  {int(user):[] for user in set(raw_data[:,0])}
    new_data = []
    for i in range(raw_data.shape[0]):
        if int(raw_data[i,0]) in user_dic:
            new_data.append([int(raw_data[i,0]),int(raw_data[i,1]),raw_data[i,2],0])

    new_data = np.array(new_data)

    for i in range(new_data.shape[0]):
        sequence_dic[int(new_data[i,0])].append([i,int(new_data[i,1]),new_data[i,2]])
    
    for user in sequence_dic.keys():
        cur_test = int(test_data_ratio*len(sequence_dic[user]))
        for i in range(cur_test):
            interaction = sequence_dic[user].pop()
            new_data[interaction[0],3] = 1

    return new_data

def train_test_generator(data, look_back):

    train,test = [],[]
    unique_users = set(data[:,0])
    items_per_user = {int(user):[0 for i in range(look_back)] for user in unique_users}
    
    for (idx,row) in enumerate(data):
        user,item,time = int(row[0]),int(row[1]),row[2]
        items_per_user[user] = items_per_user[user][1:]+[item+1]
        current_items = items_per_user[user]
        if row[3]==0:
            train.append([current_items[:-1],current_items[-1]])                                                                                            
        else:
            test.append([current_items[:-1],current_items[-1]])
                                                                
    return train,test


