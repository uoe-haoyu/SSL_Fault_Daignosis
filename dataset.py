import torch
import numpy as np
from torch.utils.data import Dataset
from SSL.dataset.Augument import Preprocess62to2s
from torchvision import transforms



class MyDataset(Dataset):
    """
    制作数据集
    """

    def __init__(self, csv_path, transform=None, loader=None, is_val=False, train=True):

        super(MyDataset, self).__init__()
        if csv_path is not None:
            df_train = np.load(csv_path)
        else:
            df_train = None

        self.train = train
        self.df = df_train
        self.loader = loader
        if csv_path is not None:
            print('当前数据集长度为：', self.__len__())
        self.T =transforms.Compose([
            Preprocess62to2s(),
        ])


    def __getitem__(self, index):

        input = self.df['data']
        label = self.df['label']

        input = input[index,:,:]
        input = self.T(input)
        label = label[index]



        input = torch.Tensor(input)
        label = torch.tensor([label], dtype=torch.long)
        label = label.squeeze(0)


        return input, label

    def __len__(self):
        if self.df is not None:
            return (self.df['data'].shape[0])


def get_data(train_path, test_path, rate=0.1, is_val=False):

    traindata = MyDataset(train_path,train=False)
    testdata = MyDataset(test_path,train=False)

    if is_val:
        valiation = MyDataset(is_val,train=False)

        # print(r'训练集占比{}, 将数据集分割为 train:{} test:{} val:{}'.format((testdata.__len__())/(traindata.__len__()+testdata.__len__()+valiation.__len__()), traindata.__len__(),testdata.__len__(), valiation.__len__()))
        return {'train': traindata, 'test': testdata, 'val':valiation}

    else:
        print('没有分割验证集合，只有训练集和测试机')
        return {'train': traindata, 'test': testdata}

def get_pathdata(test_path):
    return MyDataset(test_path,train=False)


if __name__ == '__main__':
    import random


