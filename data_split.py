import numpy as np
import random
import pdb
from tqdm import tqdm

file = 'gowalla.txt'

ls = []

with open(file, 'r') as f:
    lines = f.readlines()
    for line in lines:
        line = line.strip().split(' ')
        user = int(line[0])
        item = int(line[1])
        rating = int(line[2])

        ls.append([user, item, rating])
data = np.array(ls)

def core_filter(data, core = 5):
    while True:
        mask = []
        mask_user = np.bincount(data[:, 0]) > core - 1
        mask_item = np.bincount(data[:, 1]) > core - 1

        for line in data:
            user = line[0]
            item = line[1]
            mask.append(mask_user[user] and mask_item[item])

        data = data[mask]
        if sum(mask) == len(mask):
            break
        print(data.shape)
    data = np.unique(data, axis = 0)
    print("Uniqie data points", data.shape)

    return data

def re_mapping(data):
    user_index = 0
    item_index = 0
    user_mapping = {}
    item_mapping = {}
    users = data[:, 0]
    items = data[:, 1]

    for user in users:
        if user not in user_mapping:
            user_mapping[user] = user_index
            user_index += 1

    for item in items:
        if item not in item_mapping:
            item_mapping[item] = item_index
            item_index += 1

    for i in range(len(data)):
        data[i, 0] = user_mapping[data[i, 0]]
        data[i, 1] = item_mapping[data[i, 1]]
    
    print("Total data samples: ", len(data))
    print("User numbers: ", user_index)
    print("Item numbers: ", item_index)
    return data

def split_data(data, train_ratio = 0.8):
    np.random.shuffle(data)
    user_dic = {}
    for line in data:
        user, item, rating = line
        if user in user_dic:
            user_dic[user].append([item, rating])
        else:
            user_dic[user] = [[item, rating]]

    train_data = []
    val_data = []
    test_data = []
    val_ratio = (1 - train_ratio) / 2

    for user in tqdm(user_dic):
        interactions = user_dic[user]
        for i in range(int(len(interactions) * train_ratio)):
            item, rating = interactions[i]
            train_data.append([user, item, rating])

        for i in range(int(len(interactions) * train_ratio), int(len(interactions) * (train_ratio + val_ratio))):
            item, rating = interactions[i]
            val_data.append([user, item, rating])
        
        for i in range(int(len(interactions) * (train_ratio + val_ratio)), len(interactions)):
            item, rating = interactions[i]
            if rating > 3:
                test_data.append([user, item, rating])

    return train_data, val_data, test_data 

def write_file(data, file_name):
    with open(file_name, 'w') as f:
        for line in data:
            user = line[0]
            item = line[1]
            rating = line[2]
            f.write(str(user) + ',' + str(item) + ',' + str(rating))
            f.write('\n')
    
data = core_filter(data, core = 5)
data = re_mapping(data)
train, val, test = split_data(data)

write_file(train, './train.txt')
write_file(val, './val.txt')
write_file(test, './test.txt')



