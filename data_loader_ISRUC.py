import torch
from torch.utils.data import DataLoader
from torch.utils.data import Dataset
import os
import numpy as np
# from augmentations import DataTransform


class kFoldGenerator():
    '''
    Data Generator
    '''
    k = -1      # the fold number
    x_list = [] # x list with length=k
    y_list = [] # x list with length=k

    # Initializate
    def __init__(self, x, y, args):
        if len(x) != len(y):
            assert False, 'Data generator: Length of x or y is not equal to k.'
        self.k = len(x)
        self.x_list = x
        self.y_list = y
        self.args = args
    # Get i-th fold
    def getFold(self, i):
        isFirst = True
        for p in range(self.k):
            if p != i:
                if isFirst:
                    train_data = self.x_list[p]
                    train_targets = self.y_list[p]
                    isFirst = False
                else:
                    train_data = np.concatenate((train_data, self.x_list[p]))
                    train_targets = np.concatenate((train_targets, self.y_list[p]))
            else:
                val_data = self.x_list[p]
                val_targets = self.y_list[p]

        train_targets = np.argmax(train_targets, -1)
        val_targets = np.argmax(val_targets, -1)
        print(np.shape(train_data))

        train = Load_Dataset(train_data, train_targets, self.args)
        test = Load_Dataset(val_data, val_targets, self.args)
        val = Load_Dataset(val_data, val_targets, self.args)
        train_loader = torch.utils.data.DataLoader(dataset=train, batch_size=self.args.batch_size,
                                                   shuffle=True, drop_last=self.args.drop_last,
                                                   num_workers=0)
        valid_loader = torch.utils.data.DataLoader(dataset=val, batch_size=self.args.batch_size,
                                                   shuffle=False, drop_last=self.args.drop_last,
                                                   num_workers=0)
        test_loader = torch.utils.data.DataLoader(dataset=test, batch_size=self.args.batch_size,
                                                  shuffle=False, drop_last=False,
                                                  num_workers=0)

        return train_loader, valid_loader, test_loader

    # Get all data x
    def getX(self):
        All_X = self.x_list[0]
        for i in range(1, self.k):
            All_X = np.append(All_X, self.x_list[i], axis=0)
        return All_X

    # Get all label y (one-hot)
    def getY(self):
        All_Y = self.y_list[0][2:-2]
        for i in range(1, self.k):
            All_Y = np.append(All_Y, self.y_list[i][2:-2], axis=0)
        return All_Y

    # Get all label y (int)
    def getY_int(self):
        All_Y = self.getY()
        return np.argmax(All_Y, axis=1)


class Load_Dataset(Dataset):
    # Initialize your data, download, etc.
    def __init__(self, X_train,y_train, args):
        super(Load_Dataset, self).__init__()

        # X_train = dataset["samples"]
        # y_train = dataset["labels"]
        # print(np.shape(X_train))
        # print(np.shape(y_train))

        # print(X_train.shape)
        if len(X_train.shape) < 3:
            X_train = X_train.unsqueeze(2)

        if X_train.shape.index(min(X_train.shape)) != 1:  # make sure the Channels in second dim
            X_train = X_train.permute(0, 2, 1)

        if isinstance(X_train, np.ndarray):
            self.x_data = torch.from_numpy(X_train).float()
            self.y_data = torch.from_numpy(y_train).long()
        else:
            self.x_data = X_train.float()
            self.y_data = y_train.long()

        # print(self.x_data.size())
        # print(self.y_data.size())

        self.len = X_train.shape[0]
        # print(self.len)
        shape = self.x_data.size()
        self.x_data = self.x_data.reshape(shape[0],shape[1],args.time_denpen_len, args.window_size)
        self.x_data = torch.transpose(self.x_data, 1,2)

        # self.x_data = self.x_data.float()
        # self.y_data = self.y_data.long()

        # print(self.x_data.size())
        # print(self.y_data.size())

    # def __getitem__(self, index):
    #     return self.x_data[index], self.y_data[index], self.x_data[index], self.x_data[index]

    def __getitem__(self, index):
        return self.x_data[index], self.y_data[index]

    def __len__(self):
        return self.len


# def data_generator(data_path, args):
#
#     # train_dataset = torch.load(os.path.join(data_path, "train.pt"))
#     # valid_dataset = torch.load(os.path.join(data_path, "val.pt"))
#     # test_dataset = torch.load(os.path.join(data_path, "test.pt"))
#
#     # train_dataset = Load_Dataset(train_dataset, args)
#     # valid_dataset = Load_Dataset(valid_dataset, args)
#     # test_dataset = Load_Dataset(test_dataset, args)
#
#     train_loader = torch.utils.data.DataLoader(dataset=train_dataset, batch_size=args.batch_size,
#                                                shuffle=True, drop_last=args.drop_last,
#                                                num_workers=0)
#     valid_loader = torch.utils.data.DataLoader(dataset=valid_dataset, batch_size=args.batch_size,
#                                                shuffle=False, drop_last=args.drop_last,
#                                                num_workers=0)
#
#     test_loader = torch.utils.data.DataLoader(dataset=test_dataset, batch_size=args.batch_size,
#                                               shuffle=False, drop_last=False,
#                                               num_workers=0)
#
#     return train_loader, valid_loader, test_loader


if __name__ == '__main__':
    from args import args
    args = args()
    args.time_denpen_len = 15
    args.window_size = 200
    args.drop_last = False
    path = './ISRUC/ISRUC_S3.npz'
    ReadList = np.load(path, allow_pickle=True)
    Fold_Num = ReadList['Fold_len']  # Num of samples of each fold
    Fold_Data = ReadList['Fold_data']  # Data of each fold
    Fold_Label = ReadList['Fold_label']  # Labels of each fold

    # print(np.shape(Fold_Data[0]))
    # print(Fold_Label[0])

    print("Read data successfully")
    print('Number of samples: ', np.sum(Fold_Num))

    # ## 2.2. Build kFoldGenerator or DominGenerator
    DataGenerator = kFoldGenerator(Fold_Data, Fold_Label,args)
    train, val, test = DataGenerator.getFold(3)
    for data, label in train:
        print(data.size())
        print(label.size())