import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import Dataset
import math

class TransFLocDataset(Dataset):
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

class TransFormerLoc(nn.Module):
    def __init__(self, n_users, n_times, n_locs, n_apps, hidden_dim, dim, seq_length):
        super(TransFormerLoc, self).__init__()

        self.user_emb = nn.Embedding(n_users, dim)
        self.time_emb = nn.Embedding(n_times, dim)
        self.loc_emb = nn.Embedding(n_locs, dim)
        self.app_emb = nn.Embedding(n_apps, dim)
        self.dim = dim
        self.seq_length = seq_length

        self.input_dim = self.dim * 4
        self.hidden_dim = hidden_dim
        #self.hidden_dim = self.input_dim
        self.input_layer = nn.Linear(self.input_dim, self.hidden_dim)
        self.encoder_layer = nn.TransformerEncoderLayer(d_model=self.hidden_dim, nhead=8, batch_first=True)
        self.transformer_encoder = nn.TransformerEncoder(self.encoder_layer, num_layers=1)
        self.time_linear = nn.Linear(self.seq_length, 1)
        self.classifier = nn.Linear(self.hidden_dim, n_apps)

    def forward(self, users, times, locs, app_seq):
        # users [batch_size, 1]
        # times [batch_size, 1]
        # app_seq [batch_size, seq_length]

        batch_size = users.size(0)
        user_vector = self.user_emb(users)  # [batch_size, 1, dim]
        time_vector = self.time_emb(times)  # [batch_size, 1, dim]
        loc_vector = self.loc_emb(locs)  # [batch_size, 1, dim]
        app_seq_vector = self.app_emb(app_seq)  # [batch_size, seq_length, dim]

        # [batch_size, seq_length, input_dim]
        input_vector = torch.cat([user_vector.repeat(1, self.seq_length, 1), time_vector.repeat(1, self.seq_length, 1), loc_vector.repeat(1, self.seq_length, 1), app_seq_vector], axis=2)
        x = self.input_layer(input_vector)
        #x = input_vector
        x = self.transformer_encoder(x)
        # [B, T, H] -> [B, H, T] -> [B, H, 1]
        x = x.permute(0, 2, 1)
        x = self.time_linear(x)
        x = x.squeeze(2)
        
        #x = hidden_last.squeeze(0)
        return self.classifier(x)