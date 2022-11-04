import torch
import torch.nn as nn
import torch.nn.functional as F


def dot_graph_construction(node_features):
    ## node features size is (bs, N, dimension)
    ## output size is (bs, N, N)
    bs, N, dimen = node_features.size()

    node_features_1 = torch.transpose(node_features, 1, 2)
    Adj = torch.bmm(node_features, node_features_1)
    eyes_like = torch.eye(N).repeat(bs, 1, 1).cuda()
    eyes_like_inf = eyes_like*1e8
    Adj = F.leaky_relu(Adj-eyes_like_inf)
    Adj = F.softmax(Adj, dim = -1)
    Adj = Adj+eyes_like


    return Adj


def neighbor_information(adj, X, n_out_sensors, n_hops=3):
    adj_nhops_list = [adj]
    adj_nhops = adj
    for h in range(n_hops):
        adj_nhops = torch.bmm(adj_nhops, adj)
        adj_nhops_list.append(adj_nhops)
    adj_nhops_list = torch.stack(adj_nhops_list,0)
    adj_nhops = torch.mean(adj_nhops_list, 0)

    X_neighbor = torch.bmm(adj_nhops, X)
    X_diff = X-X_neighbor ## size is (n_sensor, features)
    X_rank = torch.mean(X_diff**2, -1)


    _, idx = torch.sort(X_rank, descending=True, dim=1)

    topk = idx[:, :n_out_sensors]
    bat_id = torch.arange(X.size(0)).unsqueeze(1)
    X_out = X[bat_id, topk]
    A_out = adj[bat_id, topk]
    A_out = torch.transpose(A_out, 1, 2)
    A_out = A_out[bat_id, topk]

    return A_out,X_out


class Feature_extractor_1DCNN(nn.Module):
    def __init__(self, input_channels, num_hidden, output_dimension, kernel_size, stride, dropout):
        super(Feature_extractor_1DCNN, self).__init__()

        self.conv_block1 = nn.Sequential(
            nn.Conv1d(input_channels, num_hidden, kernel_size=kernel_size,
                      stride=stride, bias=False, padding=(kernel_size//2)),
            nn.BatchNorm1d(num_hidden),
            nn.ReLU(),
            nn.MaxPool1d(kernel_size=2, stride=2, padding=1),
            nn.Dropout(dropout)
        )

        self.conv_block2 = nn.Sequential(
            nn.Conv1d(num_hidden, num_hidden*2, kernel_size=8, stride=1, bias=False, padding=4),
            nn.BatchNorm1d(num_hidden*2),
            nn.ReLU(),
            nn.MaxPool1d(kernel_size=2, stride=2, padding=1)
        )

        self.conv_block3 = nn.Sequential(
            nn.Conv1d(num_hidden*2, num_hidden*4, kernel_size=8, stride=1, bias=False, padding=4),
            nn.BatchNorm1d(num_hidden*4),
            nn.ReLU(),
            nn.MaxPool1d(kernel_size=2, stride=2, padding=1),
        )


    def forward(self, x_in):
        x = self.conv_block1(x_in)
        x = self.conv_block2(x)
        x = self.conv_block3(x)
        return x



class GCN_layer(nn.Module):
    def __init__(self, input_dimension, out_dimension):
        super(GCN_layer, self).__init__()

        self.nonlinear_FC = nn.Sequential(
            nn.Linear(input_dimension, out_dimension),
            nn.ReLU()
        )

    def forward(self, adj, X):
        adj_X = torch.bmm(adj, X)
        adj_X = self.nonlinear_FC(adj_X)

        return adj, adj_X





class GCN_direct(nn.Module):
    def __init__(self, input_dimension, hidden_dimension, embedding_dimension, time_length, args):
        super(GCN_direct, self).__init__()
        in_nodes = args.num_nodes
        self.args = args

        self.embedding_dimension = embedding_dimension
        time_length = args.convo_time_length
        self.time_length = args.convo_time_length
        self.Time_Preprocessing = Feature_extractor_1DCNN(input_dimension * in_nodes,
                                                                     hidden_dimension * in_nodes,
                                                                     embedding_dimension * in_nodes, 8, 1, 0.35)

        if args.layers == 1:
            self.gc1 = GCN_layer(embedding_dimension * time_length, embedding_dimension * time_length * 3)
        elif args.layers == 2:
            self.gc1 = GCN_layer(embedding_dimension * time_length, embedding_dimension * time_length* 2)
            self.gc2 = GCN_layer(embedding_dimension * time_length * 2, embedding_dimension * time_length * 3)
        elif args.layers == 3:
            self.gc1 = GCN_layer(embedding_dimension * time_length, embedding_dimension * time_length* 2)
            self.gc2 = GCN_layer(embedding_dimension * time_length * 2, embedding_dimension * time_length * 3)
            self.gc3 = GCN_layer(embedding_dimension * time_length * 3, embedding_dimension * time_length * 3)


        self.fc_0 = nn.Linear(time_length * embedding_dimension*in_nodes * 3, embedding_dimension *in_nodes* 3)
        self.fc_1 = nn.Linear(embedding_dimension*in_nodes * 3, args.num_class)

    def forward(self, X):
        ## Size of X is (bs, time_length, num_nodes, dimension)
        bs, tlen, num_node, dimension = X.size()


        X = torch.reshape(X, [bs, tlen, dimension * num_node])
        TD_input = torch.transpose(X, 2, 1)  ### size is (bs, dimension*num_node, tlen)
        TD_output = self.Time_Preprocessing(TD_input)  ### size is (bs, out_dimension*num_node, tlen)
        TD_output = torch.transpose(TD_output, 2, 1)  ### size is (bs, tlen, out_dimension*num_node

        GC_input = torch.reshape(TD_output, [bs, self.time_length, num_node, self.embedding_dimension])
        GC_input = torch.transpose(GC_input, 1, 2)

        GC_input = torch.reshape(GC_input, [bs, num_node, -1])  ## size is (bs*tlen, num_node, embedding_size)

        A_output = dot_graph_construction(GC_input)  ## size is (bs*tlen, num_node, num_node)

        if self.args.layers == 1:
            A_output, GC_output = self.gc1(A_output, GC_input)
        elif self.args.layers == 2:
            A_output_1, GC_input_1 = self.gc1(A_output, GC_input)
            A_output_2, GC_output = self.gc2(A_output_1, GC_input_1)
        elif self.args.layers == 3:
            A_output_1, GC_input_1 = self.gc1(A_output, GC_input)
            A_output_2, GC_input_2 = self.gc2(A_output_1, GC_input_1)
            A_output_2, GC_output = self.gc3(A_output_2, GC_input_2)

        GC_output = torch.reshape(GC_output, [bs, -1])
        # print(GC_output.size())
        out = F.leaky_relu(self.fc_1(F.leaky_relu(self.fc_0(GC_output))))

        return out


if __name__ == '__main__':
    from args import args
    args = args()
    args.convo_time_length = 12
    args.num_sensor_last = 6
    args.hidden_node1 = 8
    bs = 10
    n_sensors = 14
    n_features = 6
    time_length = 5
    args.num_nodes = n_sensors
    # adj = torch.rand(bs, n_sensors,n_sensors)
    features = torch.rand(bs, time_length, n_sensors, n_features).cuda()

    net = GCN_direct(n_features,10, 10, time_length,args).cuda()
    y = net(features)
    print(y.size())