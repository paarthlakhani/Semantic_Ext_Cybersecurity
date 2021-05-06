import torch.nn as nn


class BiLSTMModel(nn.Module):
    def __init__(self, in_features, hidden_features, num_reccurent=1, dropout_prob=0):  # 0.01
        #super().__init__()
        super(BiLSTMModel, self).__init__()
        self.lstm_model = nn.LSTM(input_size=in_features, hidden_size=hidden_features, num_layers=num_reccurent,
                                  bias=False,
                                  dropout=dropout_prob, bidirectional=True)

        self.linear = nn.Linear(in_features=2*hidden_features, out_features=7)
        self.dropout = nn.Dropout(dropout_prob)
        #model.loss_criterion.reduction = 'sum'
        self.loss_criterion = nn.CrossEntropyLoss(reduction='none')  # reduction='none'

    def forward(self, x):
        inter_output = self.lstm_model(x)
        output = self.linear(self.dropout(inter_output[0]))
        return output


bi = BiLSTMModel(300, 100)
