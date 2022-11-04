import torch
import torch.nn as nn



class CNN1(nn.Module):
    def __init__(self, out_dimen):
        super(CNN1, self).__init__()

        self.cnn_layers = nn.Sequential(
            # Defining a 2D convolution layer
            nn.Conv1d(1, 8, kernel_size=2, stride=1),
            nn.BatchNorm1d(8),
            nn.ReLU(inplace=True),
            nn.AvgPool1d(kernel_size=2, stride=2),
            # Defining another 2D convolution layer
            nn.Conv1d(8, out_dimen, kernel_size=2, stride=1),
            nn.BatchNorm1d(out_dimen),
            nn.ReLU(inplace=True)
        )


    # Defining the forward pass
    def forward(self, x):
        # print(x.size())
        bs, time_length, num_sensors = x.size()

        x = torch.reshape(x, (bs*time_length, num_sensors))
        x = torch.unsqueeze(x, 1)
        x = self.cnn_layers(x)
        x = torch.reshape(x, (bs*time_length, -1))
        x = torch.reshape(x, (bs, time_length, -1))

        return x


import torch
import numpy as np

from torch import nn
from torch import optim

from torch.autograd import Variable
import torch.nn.functional as F



class Encoder(nn.Module):
    """encoder in DA_RNN."""

    def __init__(self, Time_length,
                 input_size,
                 encoder_num_hidden):
        """Initialize an encoder in DA_RNN."""
        super(Encoder, self).__init__()
        self.encoder_num_hidden = encoder_num_hidden
        self.input_size = input_size
        self.T = Time_length

        # Fig 1. Temporal Attention Mechanism: Encoder is LSTM
        self.encoder_lstm = nn.LSTM(
            input_size=self.input_size,
            hidden_size=self.encoder_num_hidden,
            num_layers=1
        )

        # Construct Input Attention Mechanism via deterministic attention model
        # Eq. 8: W_e[h_{t-1}; s_{t-1}] + U_e * x^k
        self.encoder_attn = nn.Linear(
            in_features=2 * self.encoder_num_hidden + self.T,
            out_features=1
        )

    def forward(self, X):
        """forward.

        Args:
            X: input data

        """
        X_tilde = Variable(X.data.new(
            X.size(0), self.T, self.input_size).zero_())
        X_encoded = Variable(X.data.new(
            X.size(0), self.T, self.encoder_num_hidden).zero_())


        h_n = self._init_states(X)
        s_n = self._init_states(X)

        for t in range(self.T):
            x = torch.cat((h_n.repeat(self.input_size, 1, 1).permute(1, 0, 2),
                           s_n.repeat(self.input_size, 1, 1).permute(1, 0, 2),
                           X.permute(0, 2, 1)), dim=2)

            x = self.encoder_attn(
                x.view(-1, self.encoder_num_hidden * 2 + self.T))

            # get weights by softmax
            alpha = F.softmax(x.view(-1, self.input_size), dim=1)

            # get new input for LSTM
            x_tilde = torch.mul(alpha, X[:, t, :])

            # Fix the warning about non-contiguous memory
            self.encoder_lstm.flatten_parameters()

            # encoder LSTM
            _, final_state = self.encoder_lstm(
                x_tilde.unsqueeze(0), (h_n, s_n))
            h_n = final_state[0]
            s_n = final_state[1]

            X_tilde[:, t, :] = x_tilde
            X_encoded[:, t, :] = h_n

        return X_tilde, X_encoded

    def _init_states(self, X):
        """Initialize all 0 hidden states and cell states for encoder."""
        # https://pytorch.org/docs/master/nn.html?#lstm
        return Variable(X.data.new(1, X.size(0), self.encoder_num_hidden).zero_())


class Decoder(nn.Module):
    """decoder in DA_RNN."""

    def __init__(self, Time_length, decoder_num_hidden, encoder_num_hidden, num_class):
        """Initialize a decoder in DA_RNN."""
        super(Decoder, self).__init__()
        self.decoder_num_hidden = decoder_num_hidden
        self.encoder_num_hidden = encoder_num_hidden
        self.T = Time_length

        self.attn_layer = nn.Sequential(
            nn.Linear(2 * decoder_num_hidden +
                      encoder_num_hidden, encoder_num_hidden),
            nn.Tanh(),
            nn.Linear(encoder_num_hidden, 1)
        )
        self.lstm_layer = nn.LSTM(
            input_size=1,
            hidden_size=decoder_num_hidden
        )
        self.fc = nn.Linear(encoder_num_hidden, 1)
        self.fc_final = nn.Linear(decoder_num_hidden + encoder_num_hidden, num_class)

        self.fc.weight.data.normal_()

    def forward(self, X_encoded):
        """forward."""
        d_n = self._init_states(X_encoded)
        c_n = self._init_states(X_encoded)

        for t in range(self.T):

            x = torch.cat((d_n.repeat(self.T, 1, 1).permute(1, 0, 2),
                           c_n.repeat(self.T, 1, 1).permute(1, 0, 2),
                           X_encoded), dim=2)

            beta = F.softmax(self.attn_layer(
                x.view(-1, 2 * self.decoder_num_hidden + self.encoder_num_hidden)).view(-1, self.T), dim=1)

            # Eqn. 14: compute context vector
            # batch_size * encoder_hidden_size
            context = torch.bmm(beta.unsqueeze(1), X_encoded)[:, 0, :]
            if t < self.T:
                # Eqn. 15
                # batch_size * 1
                # y_tilde = self.fc(
                #     torch.cat((context, y_prev[:, t].unsqueeze(1)), dim=1))
                y_tilde = self.fc(context)
                # Eqn. 16: LSTM
                self.lstm_layer.flatten_parameters()
                _, final_states = self.lstm_layer(
                    y_tilde.unsqueeze(0), (d_n, c_n))

                d_n = final_states[0]  # 1 * batch_size * decoder_num_hidden
                c_n = final_states[1]  # 1 * batch_size * decoder_num_hidden

        # Eqn. 22: final output
        y_pred = self.fc_final(torch.cat((d_n[0], context), dim=1))

        return y_pred

    def _init_states(self, X):
        """Initialize all 0 hidden states and cell states for encoder."""
        # hidden state and cell state [num_layers*num_directions, batch_size, hidden_size]
        # https://pytorch.org/docs/master/nn.html?#lstm
        return Variable(X.data.new(1, X.size(0), self.decoder_num_hidden).zero_())





class DABi_LSTM(nn.Module):
    def __init__(self, num_sensors,time_length, encoder_hidden_dim, decoder_hidden_dim, num_class):
        super(DABi_LSTM, self).__init__()

        self.encoder = Encoder(time_length, num_sensors, encoder_hidden_dim)

        self.decoder = Decoder(time_length, decoder_hidden_dim, encoder_hidden_dim, num_class)



    def forward(self, X):
        ## X size is (bs, time_length, num_sensors)
        bs, time_length, num_sensors = X.size()

        input_weighted, input_encoded = self.encoder(X)
        y_pred = self.decoder(input_encoded)


        return y_pred

# class Bi_LSTM(nn.Module):
#     def __init__(self, input_dim, output_dim):
#         super(Bi_LSTM, self).__init__()
#         self.atten = nn.Sequential(nn.Linear(input_dim, input_dim),
#                                    nn.Tanh(),
#                                    nn.Linear(input_dim, 1))
#
#         self.bi_lstm1 = nn.LSTM(input_size=input_dim,
#                                 hidden_size=output_dim,
#                                 num_layers=1,
#                                 batch_first=True,
#                                 dropout=0,
#                                 bidirectional=False)
#
#
#     # Defining the forward pass
#     def forward(self, x):
#         x = torch.transpose(x,1,0)
#
#         weights = self.atten(x)
#         print(weights.size())
#         x, hidden = self.bi_lstm1(x)
#         print(x.size())
#
#         x = x*weights
#
#
#         # x_split = torch.split(x, (x.shape[2] // 2), 2)
#         # x = x_split[0] + x_split[1]
#         # x, hidden = self.bi_lstm2(x)
#         # x_split = torch.split(x, (x.shape[2] // 2), 2)
#         # x = x_split[0] + x_split[1]
#         # x = torch.transpose(x,1,0)
#
#         # # concat two inputs
#         # x2 = torch.cat((x, aux), dim=2)
#         # x2, hidden = self.bi_lstm3(x2)
#         # x2_presp = torch.split(x2, 1, 1)
#         # x2_split = torch.split(x2_presp, (x2_presp.shape[2] // 2), 2)
#         # x2 = x2_split[0] + x2_split[1]
#         # x2 = self.drop3(x2)
#
#         # x = x.reshape(x.shape[0], -1)
#         # x = self.fc(x)
#         # out = self.cls(x)
#
#         return x







class net(nn.Module):
    def __init__(self, out_features, conv_sensors,time_length, encoder_hidden_dim, decoder_hidden_dim, num_class):
        super(net, self).__init__()

        self.cnn1 = CNN1(out_features)

        self.encoder = Encoder(time_length, out_features * conv_sensors, encoder_hidden_dim)

        self.decoder = Decoder(time_length, decoder_hidden_dim, encoder_hidden_dim, num_class)


    def forward(self, X):
        ## X size is (bs, time_length, num_sensors)
        bs, time_length, num_sensors = X.size()

        X = self.cnn1(X)
        input_weighted, input_encoded = self.encoder(X)
        y_pred = self.decoder(input_encoded)
        return y_pred
if __name__ == '__main__':
    bs, time_length, feature_dimension = 30, 50, 20
    x = torch.rand(bs, time_length, feature_dimension)

    net = net(3, 8, time_length,20, 20, 6)

    y = net(x)

    print(y.size())