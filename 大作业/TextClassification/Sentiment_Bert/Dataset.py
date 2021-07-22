import json
import random

import pandas as pd
import numpy as np
import torch.utils.data as Data
import torch


# 自定义dataloader
class myDataset(Data.Dataset):

    def __init__(self, user_tensor, item_tensor, target_tensor):
        self.user_tensor = user_tensor
        self.item_tensor = item_tensor
        self.target_tensor = target_tensor

    def __getitem__(self, index):
        return self.user_tensor[index], self.item_tensor[index], self.target_tensor[index]

    def __len__(self):
        return self.user_tensor.size(0)


# 输入训练集数据和预测集数据，生成userId对应的Index
def generate_user_index(train_rating, test_rating):
    # 对uid和mid重新索引
    train_user_id = train_rating[['user_id']].drop_duplicates()
    test_user_id = test_rating[['user_id']].drop_duplicates()
    user_id = pd.concat([train_user_id, test_user_id]).drop_duplicates()
    # print(user_id.user_id.size, user_id.user_id.unique().size)
    user_id['user_index'] = np.arange(len(user_id))
    user_id.to_csv("./Data/rec_class/user_index.csv", index=None)


# 输入训练集数据和预测集数据，生成itemId对应的Index
def generate_item_index(train_data, test_rating):
    train_item_id = train_data[['click_article_id']].drop_duplicates()
    test_item_id = test_rating[['click_article_id']].drop_duplicates()
    item_id = pd.concat([train_item_id, test_item_id]).drop_duplicates()
    # print(item_id.click_article_id.size, item_id.click_article_id.unique().size)
    item_id['item_index'] = np.arange(len(item_id))
    item_id.to_csv("./Data/rec_class/item_index.csv", index=None)

# 预处理训练数据
def preprocess_train_data(train_rating, train_negative):
    # assert 'click_article_id' in train_rating.columns
    # assert 'user_id' in train_rating.columns

    user_index = pd.read_csv("./Data/rec_class/user_index.csv")
    train_data = pd.merge(train_rating, user_index, on=['user_id'], how='left')
    train_negative = pd.merge(train_negative, user_index, on='user_id', how='left')

    item_index = pd.read_csv("./Data/rec_class/item_index.csv")
    train_data = pd.merge(train_data, item_index, on=['click_article_id'], how='left')
    train_negative.rename(columns={'sampled_negative_item': 'click_article_id'}, inplace=True)
    train_negative = pd.merge(train_negative, item_index, on='click_article_id', how='left')

    train_data = train_data[['user_id', 'user_index', 'click_article_id', 'item_index', 'ratings']]
    print('Range of userId is [{}, {}]'.format(user_index.user_index.min(), user_index.user_index.max()))
    print('Range of itemId is [{}, {}]'.format(item_index.item_index.min(), item_index.item_index.max()))
    # print(test_data)

    return train_data, train_negative

# 预处理预测数据
def preprocess_predict_data(test_rating):
    user_index = pd.read_csv("./Data/rec_class/user_index.csv")
    item_index = pd.read_csv("./Data/rec_class/item_index.csv")
    test_data = pd.merge(test_rating, user_index, on=['user_id'], how='left')
    test_data = pd.merge(test_data, item_index, on=['click_article_id'], how='left')
    return test_data

# 负采样
def downSample(data):
    interact_status = data.groupby('user_id')['click_article_id'].apply(set).reset_index().rename(
        columns={'itemId': 'interacted_items'})

    interact_status['none_click_article_id'] = interact_status['click_article_id'].apply(
        lambda x: list(set(data['click_article_id'].unique()) - x))

    interact_status['click_article_id'] = interact_status['click_article_id'].apply(lambda x: list(x))

    interact_status.to_csv("./Data/rec_class/down_sample.csv", index=None)

# 生成负样本
def generate_train_negative(negativeData, nums_negative):
    negativeData['augment_negative'] = negativeData['none_click_article_id'].apply(
        lambda x: random.sample(json.loads(x), k=nums_negative))
    negative_init = negativeData[['user_id', 'augment_negative']].copy()
    sampled_negative = None
    for k in range(nums_negative):
        negative_init['sampled_negative_item'] = negative_init['augment_negative'].apply(lambda x: x[k])
        tem = negative_init[['user_id', 'sampled_negative_item']]
        tem['ratings'] = 0.
        sampled_negative = pd.concat([sampled_negative, tem])
    sampled_negative.to_csv("./Data/rec_class/train_negative.csv", index=None)


if __name__ == '__main__':
    # 1. 读取训练集和预测集
    trainData = pd.read("大作业/datasets/IMDB/aclImdb/train/neg")

#
#     # 2. 生成所有user和item的索引文件
#     generate_user_index(trainData, predictData)
#     generate_item_index(trainData, predictData)
#
#     # 3. 为训练集负采样
#     downSample(trainData)
#     downSample = pd.read_csv("./Data/rec_class/down_sample.csv")
#     generate_train_negative(downSample, 30)
#
#     # 预处理数据
#     train_negative = pd.read_csv("./Data/rec_class/train_negative.csv")
#     trainData, train_negative = preprocess_train_data(trainData, train_negative)
#
#     # 4. 为训练集加入负样本
#     trainData = pd.concat([trainData, train_negative])
#
#     # 样本划分测试
#     # 划分训练集测试集
#     train_size = int(len(trainData) * 0.7)
#     test_size = len(trainData) - train_size
#
#     users, items, target = trainData['user_index'].tolist(), trainData[
#         'item_index'].tolist(), trainData['ratings'].tolist()
#     datasets = myDataset(user_tensor=torch.LongTensor(users),
#                          item_tensor=torch.LongTensor(items),
#                          target_tensor=torch.LongTensor(target))
#     train_datasets, test_datasets = torch.utils.data.random_split(datasets, [train_size, test_size])
#
#     dataloader = Data.DataLoader(dataset=train_datasets, batch_size=4, num_workers=2)
