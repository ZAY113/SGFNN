import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import Dataset
import math

class DeepAppDataset(Dataset):
    def __init__(self, df):
        self.df = df

    def __len__(self):
        return len(self.df)

    def __getitem__(self, idx):
        user = self.df.iloc[idx]['user']
        time = self.df.iloc[idx]['time']
        loc = self.df.iloc[idx]['location']
        app = self.df.iloc[idx]['app']
        app_seq = self.df.iloc[idx]['app_seq']
        return (torch.LongTensor([user]), torch.LongTensor([time]), torch.LongTensor([loc]), torch.LongTensor([app]), torch.LongTensor(app_seq))

class DeepApp(nn.Module):
    def __init__(self, n_users, n_times, n_locs, n_apps, hidden_dim, dim, seq_length, model_type):
        super(DeepApp, self).__init__()

        self.time_emb = nn.Embedding(n_times, dim)
        self.loc_emb = nn.Embedding(n_locs, dim)
        self.app_emb = nn.Embedding(n_apps, dim)
        self.dim = dim
        self.seq_length = seq_length
        self.model_type = model_type

        self.input_dim = self.dim * 3
        self.hidden_dim = hidden_dim
        self.scale = 1. / math.sqrt(self.hidden_dim)
        if self.model_type == "LSTM":
            self.rnn = nn.LSTM(self.input_dim, self.hidden_dim, num_layers=1, batch_first=True)
        elif self.model_type == "GRU":
            self.rnn = nn.GRU(self.input_dim, self.hidden_dim, num_layers=1, batch_first=True)
        else:
            self.rnn = nn.RNN(self.input_dim, self.hidden_dim, num_layers=1, batch_first=True)
        self.user_classifier = nn.Linear(self.hidden_dim, n_users)
        self.loc_classifier = nn.Linear(self.hidden_dim, n_locs)
        self.app_classifier = nn.Linear(self.hidden_dim, n_apps)

    def forward(self, times, locs, app_seq):
        # times [batch_size, 1]
        # locs [batch_size, 1]
        # app_seq [batch_size, seq_length]

        batch_size = times.size(0)
        time_vector = self.time_emb(times)  # [batch_size, 1, dim]
        loc_vector = self.loc_emb(locs)  # [batch_size, 1, dim]
        app_seq_vector = self.app_emb(app_seq)  # [batch_size, seq_length, dim]

        input_vector = torch.cat([time_vector.repeat(1, self.seq_length, 1), loc_vector.repeat(1, self.seq_length, 1), app_seq_vector], axis=2)    # [batch_size, seq_length, input_dim]
        if self.model_type == "LSTM":
            output, (hidden_last, cell_last) = self.rnn(input_vector)
        else:
            output, hidden_last = self.rnn(input_vector)
        
        x = hidden_last.permute(1, 0, 2)    # [batch_size, 1, hidden_dim]
        x = x.squeeze(1)

        
        #x = hidden_last.squeeze(0)
        return self.user_classifier(x), self.loc_classifier(x), self.app_classifier(x)