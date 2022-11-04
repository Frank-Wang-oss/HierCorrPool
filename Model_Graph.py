import torch as tr
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
import numpy as np
import matplotlib.pyplot as plt
import math
from collections import OrderedDict
import Model_Base



class Graph_Classification(nn.Module):
    def __init__(self, input_dimension, hidden_dimension, embedding_dimension, time_length, args):
        super(Graph_Classification, self).__init__()
        in_nodes = args.num_nodes
        self.args = args

        self.embedding_dimension = args.embedding_dimension
        time_length = args.convo_time_length
        self.time_length = args.convo_time_length
        self.Time_Preprocessing = Model_Base.Feature_extractor_1DCNN(input_dimension * in_nodes,
                                                                     hidden_dimension * in_nodes,
                                                                     embedding_dimension * in_nodes, 8, 1, 0.35)

        if args.layers == 1:
            self.gc1 = Model_Base.Graph_Classification_block(embedding_dimension * time_length, embedding_dimension * time_length * 3,
                                       in_nodes, args.num_sensor_last, args)
        elif args.layers == 2:
            self.gc1 = Model_Base.Graph_Classification_block(embedding_dimension * time_length, embedding_dimension * time_length* 2,
                                                             in_nodes, args.hidden_node1, args)
            self.gc2 = Model_Base.Graph_Classification_block(embedding_dimension * time_length * 2, embedding_dimension * time_length * 3,
                                                             args.hidden_node1, args.num_sensor_last, args)
        elif args.layers == 3:
            self.gc1 = Model_Base.Graph_Classification_block(embedding_dimension * time_length, embedding_dimension * time_length* 2,
                                                             in_nodes, args.hidden_node1, args)
            self.gc2 = Model_Base.Graph_Classification_block(embedding_dimension * time_length * 2, embedding_dimension * time_length * 3,
                                                             args.hidden_node1, args.hidden_node2, args)
            self.gc3 = Model_Base.Graph_Classification_block(embedding_dimension * time_length * 3, embedding_dimension * time_length * 3,
                                                             args.hidden_node2, args.num_sensor_last, args)


        self.fc_0 = nn.Linear(time_length * args.num_sensor_last * embedding_dimension * 3, embedding_dimension * 3)
        self.fc_1 = nn.Linear(embedding_dimension * 3, args.num_class)

    def forward(self, X):

        bs, tlen, num_node, dimension = X.size()

        X = tr.reshape(X, [bs, tlen, dimension * num_node])
        TD_input = tr.transpose(X, 2, 1)  ### size is (bs, dimension*num_node, tlen)
        TD_output = self.Time_Preprocessing(TD_input)  ### size is (bs, out_dimension*num_node, tlen)
        TD_output = tr.transpose(TD_output, 2, 1)  ### size is (bs, tlen, out_dimension*num_node

        GC_input = tr.reshape(TD_output, [bs, self.time_length, num_node, -1])
        GC_input = tr.transpose(GC_input, 1, 2)

        GC_input = tr.reshape(GC_input, [bs, num_node, -1])  ## size is (bs*tlen, num_node, embedding_size)

        A_output = Model_Base.dot_graph_construction(GC_input)  ## size is (bs*tlen, num_node, num_node)

        if self.args.layers == 1:
            A_output, GC_output = self.gc1(A_output, GC_input)
        elif self.args.layers == 2:
            A_output_1, GC_input_1 = self.gc1(A_output, GC_input)
            A_output_2, GC_output = self.gc2(A_output_1, GC_input_1)
        elif self.args.layers == 3:
            A_output_1, GC_input_1 = self.gc1(A_output, GC_input)
            A_output_2, GC_input_2 = self.gc2(A_output_1, GC_input_1)
            A_output_2, GC_output = self.gc3(A_output_2, GC_input_2)

        GC_output = tr.reshape(GC_output, [bs, -1])

        out = F.leaky_relu(self.fc_1(F.leaky_relu(self.fc_0(GC_output))))

        return out






if __name__ == '__main__':
    window_size = 7
    time_denpen_len = 5
    num_nodes = 14

    from args import args
    args = args()
    args.data_sub = 1
    args.window_size = 7
    args.time_denpen_len = 5
    args.num_sensor_last = 6
    args.hidden_dimension = 10
    args.embedding_dimension = 10
    args.num_nodes = 14
    args.hidden_node1 = 10
    args.hidden_node2 = 8
    args.num_class = 1
    args.convo_time_length = 12

    X = tr.rand(30, time_denpen_len, num_nodes, window_size).cuda()
    net = Graph_Classification(input_dimension=window_size,
                             hidden_dimension=window_size,
                             embedding_dimension=window_size,
                             time_length=time_denpen_len,
                             args=args)

    net = net.cuda()
    Y = net(X)
    print(Y.size())