import torch
import torch.nn as nn
import torch.nn.functional as F








class CNN_LSTM(nn.Module):
    def __init__(self, in_dimen, conv_length, time_length, num_class):
        super(CNN_LSTM, self).__init__()
        self.CNN1_1 = nn.Sequential(nn.Conv1d(in_dimen, 128, 9),
                                    nn.ReLU())
        self.CNN1_2 = nn.Sequential(nn.Conv1d(in_dimen, 128, 7),
                                    nn.ReLU())
        self.CNN1_3 = nn.Sequential(nn.Conv1d(in_dimen, 128, 5),
                                    nn.ReLU())
        self.CNN1_4 = nn.Sequential(nn.Conv1d(in_dimen, 128, 3),
                                    nn.ReLU())

        self.Mpool_1 = nn.MaxPool1d(3)


        self.CNN2_1 = nn.Sequential(nn.Conv1d(128, 64, 9),
                                    nn.ReLU())
        self.CNN2_2 = nn.Sequential(nn.Conv1d(128, 64, 7),
                                    nn.ReLU())
        self.CNN2_3 = nn.Sequential(nn.Conv1d(128, 64, 5),
                                    nn.ReLU())
        self.CNN2_4 = nn.Sequential(nn.Conv1d(128, 64, 3),
                                    nn.ReLU())

        self.Mpool_2 = nn.MaxPool1d(3)

        self.fc1 = nn.Sequential(nn.Linear(conv_length*64, 512),
                                 nn.ReLU())

        self.bi_lstm1 = nn.LSTM(input_size=in_dimen,
                                hidden_size=27,
                                num_layers=1,
                                batch_first=True,
                                dropout=0,
                                bidirectional=True)

        self.fc2 = nn.Sequential(nn.Linear(time_length*27*2, 512),
                                 nn.ReLU())

        self.fc3 = nn.Sequential(nn.Linear(1024, num_class),
                                 nn.ReLU())
    def forward(self, x):
        x_CNN = torch.transpose(x, -1,-2)
        x_lstm = x

        x1_1 = self.CNN1_1(x_CNN)
        x1_2 = self.CNN1_2(x_CNN)
        x1_3 = self.CNN1_3(x_CNN)
        x1_4 = self.CNN1_4(x_CNN)

        x_CNN = torch.cat((x1_1, x1_2, x1_3, x1_4), -1)
        x_CNN = self.Mpool_1(x_CNN)

        x2_1 = self.CNN2_1(x_CNN)
        x2_2 = self.CNN2_2(x_CNN)
        x2_3 = self.CNN2_3(x_CNN)
        x2_4 = self.CNN2_4(x_CNN)

        x_CNN = torch.cat((x2_1, x2_2, x2_3, x2_4), -1)
        x_CNN = self.Mpool_1(x_CNN)

        x_CNN = torch.reshape(x_CNN, (x_CNN.size(0), -1))
        x_CNN = F.dropout(x_CNN, 0.05)
        # print(x_CNN.size(1)/64)


        x_CNN = self.fc1(x_CNN)

        x_lstm, hidden = self.bi_lstm1(x_lstm)
        # print(x_lstm.size())

        x_lstm = torch.reshape(x_lstm, (x_lstm.size(0), -1))

        x_lstm = self.fc2(x_lstm)
        x_lstm = F.dropout(x_lstm, 0.05)

        x = torch.cat((x_CNN, x_lstm), -1)

        x = self.fc3(x)

        return x

if __name__ == '__main__':
    from thop import profile

    bs, time_length, feature_dimension = 1, 35, 14
    x = torch.rand(bs, time_length, feature_dimension)

    model = CNN_LSTM(feature_dimension, 46, time_length, 1)
    y = model(x)
    print(y.size())

    flops, params = profile(model, inputs=(x,))
    print(flops)
    print(params)

# if __name__ == '__main__':
#     bs, time_length, feature_dimension = 30, 72, 9
#     x = torch.rand(bs, time_length, feature_dimension)
#
#     model = CNN_LSTM(feature_dimension, 112, time_length, 6)
#     y = model(x)
#     print(y.size())

