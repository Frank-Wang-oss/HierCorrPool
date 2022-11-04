import torch as tr
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
import numpy as np
import matplotlib.pyplot as plt
import time
import math
import os
import argparse
import matplotlib.pyplot as plt
import random
import Model
import sys
sys.path.append('../../')
import data_loader_RUL
import data_loader_HAR
import data_loader_ISRUC

class Train():
    def __init__(self, args):
        self.args = args
        if args.dataset == 'RUL':
            self.train, self.valid, self.test = data_loader_RUL.data_generator(args,'../../',
                            data_set='FD00{}'.format(args.data_sub),
                            max_rul=125,
                            seq_len=args.window_size,
                            time_denpen_len=args.time_denpen_len)

        elif args.dataset == 'HAR':
            self.train, self.valid, self.test = data_loader_HAR.data_generator('../../HAR/',args=args)

        elif args.dataset == 'ISRUC':
            path = '../../ISRUC/ISRUC_S3.npz'
            ReadList = np.load(path, allow_pickle=True)
            Fold_Num = ReadList['Fold_len']  # Num of samples of each fold
            Fold_Data = ReadList['Fold_data']  # Data of each fold
            Fold_Label = ReadList['Fold_label']  # Labels of each fold

            DataGenerator = data_loader_ISRUC.kFoldGenerator(Fold_Data, Fold_Label, args)
            self.train, self.valid, self.test = DataGenerator.getFold(args.i_ISRUC)

        # print(self.train.__len__())
        # print(self.valid.__len__())
        # print(self.test.__len__())

        self.net = Model.GCN_direct(input_dimension=args.window_size,
                                                    hidden_dimension=args.hidden_dimension,
                                                    embedding_dimension=args.embedding_dimension,
                                                    time_length=args.time_denpen_len,
                                                    args = args)

        self.net = self.net.cuda() if tr.cuda.is_available() else self.net
        self.optim = optim.Adam(self.net.parameters())
        if args.task_type =='forecasting':
            self.loss_function = nn.MSELoss()
        elif args.task_type == 'classification':
            self.loss_function = nn.CrossEntropyLoss()

    def Train_batch(self):
        self.net.train()
        loss_ = 0
        for data, label in self.train:
            data = data.cuda() if tr.cuda.is_available() else data
            label = label.cuda() if tr.cuda.is_available() else label
            self.optim.zero_grad()
            prediction = self.net(data)
            loss = self.loss_function(prediction, label)
            loss.backward()
            self.optim.step()
            loss_ = loss_ + loss.item()
        return loss_

    def Train_model(self):
        epoch = self.args.epoch
        best_cross_loss = np.inf
        best_cross_accu = 0
        test_RMSE = []
        test_score = []

        test_accu = []

        predicted = []
        real = []
        for i in range(epoch):
            loss = self.Train_batch()
            if i%self.args.show_interval == 0:
                cross_result = self.Cross_validation()
                if self.args.task_type == 'forecasting':
                    if cross_result < best_cross_loss:
                        best_cross_loss = cross_result
                        test_RMSE_, test_score_, test_result_predicted, test_result_real = self.Prediction()
                        print('In the {}th epoch, TESTING RMSE is {}, TESTING Score is {}'.format(i, test_RMSE_, test_score_))
                        test_RMSE.append(test_RMSE_)
                        test_score.append(test_score_)
                        predicted.append(test_result_predicted)
                        real.append(test_result_real)
                elif self.args.task_type == 'classification':
                    if cross_result > best_cross_accu:
                        best_cross_accu = cross_result
                        test_accu_, test_result_predicted, test_result_real = self.Prediction()
                        print('In the {}th epoch, TESTING ACCU is {}'.format(i, test_accu_))
                        test_accu.append(test_accu_)
                        predicted.append(test_result_predicted)
                        real.append(test_result_real)

        if self.args.task_type == 'forecasting':
            np.save('./experiment/{}.npy'.format(self.args.save_name),[test_RMSE, test_score, predicted, real])
        elif self.args.task_type == 'classification':
            np.save('./experiment/{}.npy'.format(self.args.save_name),[test_accu, predicted, real])



    def cuda_(self, x):
        x = tr.Tensor(np.array(x))
        if tr.cuda.is_available():
            return x.cuda()
        else:
            return x

    def data_preprocess_transpose(self, data, ops):
        '''

        :param data: size is [bs, time_length, dimension, Num_nodes]
        :return: size is [bs, time_length, Num_nodes, dimension]
        '''

        data = tr.transpose(data,2,3)
        ops = tr.transpose(ops,2,3)

        return data, ops

    def Cross_validation(self):
        self.net.eval()
        predicted_ = []
        real_ = []
        for data, label in self.valid:
            data = data.cuda() if tr.cuda.is_available() else data
            label = label.cuda() if tr.cuda.is_available() else label
            predicted = self.net(data)
            predicted_.append(predicted.detach())
            real_.append(label)
        predicted_ = tr.cat(predicted_, 0)
        real_ = tr.cat(real_, 0)
        if self.args.task_type == 'forecasting':
            MSE = self.loss_function(predicted_, real_)
            RMSE = tr.sqrt(MSE) * self.args.max_rul
            score = self.scoring_function(predicted_, real_)
            return RMSE.cpu().numpy()
        elif self.args.task_type == 'classification':
            predicted_ = tr.argmax(predicted_, -1)
            accu = self.accu_(predicted_, real_)
            return accu

    def accu_(self, predicted, real):
        num = predicted.size(0)
        correct = 0
        for i in range(num):
            if predicted[i]==real[i]:
                correct+=1

        return 100*(correct/num)

    def Prediction(self):
        '''
        This is to predict the results for testing dataset
        :return:
        '''
        self.net.eval()
        predicted_ = []
        real_ = []
        for data, label in self.test:
            data = data.cuda() if tr.cuda.is_available() else data
            label = label.cuda() if tr.cuda.is_available() else label
            predicted = self.net(data)
            predicted_.append(predicted.detach())
            real_.append(label)
        predicted_ = tr.cat(predicted_, 0)
        real_ = tr.cat(real_, 0)
        if self.args.task_type == 'forecasting':
            MSE = self.loss_function(predicted_, real_)
            RMSE = tr.sqrt(MSE) * self.args.max_rul
            score = self.scoring_function(predicted_, real_)
            return RMSE.cpu().numpy(), \
                   score.cpu().numpy(), \
                   predicted_.cpu().numpy(), \
                   real_.cpu().numpy()
        elif self.args.task_type == 'classification':
            predicted_ = tr.argmax(predicted_, -1)
            accu = self.accu_(predicted_, real_)
            return accu, \
                   predicted_.cpu().numpy(), \
                   real_.cpu().numpy()


    def visualization(self, prediction, real):
        fig = plt.figure()
        sub = fig.add_subplot(1, 1, 1)

        sub.plot(prediction, color = 'red', label = 'Predicted Labels')
        sub.plot(real, 'black', label = 'Real Labels')
        sub.legend()
        plt.show()

    def scoring_function(self, predicted, real):
        score = 0
        num = predicted.size(0)
        for i in range(num):

            if real[i] > predicted[i]:
                score = score+ (tr.exp((real[i]*self.args.max_rul-predicted[i]*self.args.max_rul)/13)-1)

            elif real[i]<= predicted[i]:
                score = score + (tr.exp((predicted[i]*self.args.max_rul-real[i]*self.args.max_rul)/10)-1)

        return score


if __name__ == '__main__':
    from args import args

    args = args()


    def args_config_RUL(data_sub, args):

        args.dataset = 'RUL'
        args.epoch = 21
        args.k = 1
        args.fe_type = '1DCNN'
        args.task_type = 'forecasting'
        args.layers = 2
        args.num_nodes = 14
        args.hidden_node1 = 10
        args.hidden_node2 = 8
        args.num_class = 1
        args.convo_time_length = 12

        if data_sub == 1:
            args.data_sub = 1
            args.window_size = 7
            args.time_denpen_len = 5
            args.num_sensor_last = 6
            args.hidden_dimension = 10
            args.embedding_dimension = 10

        if data_sub == 2:
            args.data_sub = 2
            args.window_size = 10
            args.time_denpen_len = 6
            args.num_sensor_last = 6
            args.hidden_dimension = 10
            args.embedding_dimension = 10

        if data_sub == 3:
            args.data_sub = 3
            args.window_size = 6
            args.time_denpen_len = 9
            args.num_sensor_last = 8
            args.hidden_dimension = 7
            args.embedding_dimension = 7

        if data_sub == 4:
            args.data_sub = 4
            args.window_size = 10
            args.time_denpen_len = 8
            args.num_sensor_last = 6
            args.hidden_dimension = 10
            args.embedding_dimension = 10

        return args

    def args_config_HAR(args):
        args.dataset = 'HAR'
        args.epoch = 21
        args.k = 1
        args.window_size = 8
        args.time_denpen_len = 16
        args.num_sensor_last = 4
        args.layers = 1
        args.num_nodes = 9
        args.fe_type = '1DCNN'
        args.task_type = 'classification'
        args.hidden_dimension = 8
        args.embedding_dimension = 8
        args.hidden_node1 = 7
        args.hidden_node2 = 6
        args.num_class = 6
        args.convo_time_length = 16
        return args

    def args_config_ISRUC(args):
        args.dataset = 'ISRUC'
        args.epoch = 21
        args.k = 1
        args.window_size = 10
        args.time_denpen_len = 300
        args.num_sensor_last = 2
        args.layers = 1
        args.num_nodes = 10
        args.fe_type = '1DCNN'
        args.task_type = 'classification'
        args.hidden_dimension = 10
        args.embedding_dimension = 10
        args.hidden_node1 = 8
        args.hidden_node2 = 6
        args.num_class = 5
        args.convo_time_length = 160

        return args



    for datasub in range(1, 5):
        args = args_config_RUL(datasub, args)
        for i in range(30):
            args.save_name = 'GCN_direct_RUL_datasub{}_{}'.format(datasub, i)
            train = Train(args)
            train.Train_model()
            print(args.save_name)

    args = args_config_HAR(args)
    for i in range(30):
        args.save_name = 'GCN_direct_HAR_{}'.format(i)
        train = Train(args)
        train.Train_model()
        print(args.save_name)

    args = args_config_ISRUC(args)
    for num_subject in range(10):
        args.i_ISRUC = num_subject
        for i in range(3):
            args.save_name = 'GCN_direct_{}_sub{}_{}'.format('ISRUC',num_subject, i)
            train = Train(args)
            train.Train_model()
            print(args.save_name)