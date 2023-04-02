import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import Dataset

class DNNDataset(Dataset):
    def __init__(self, df_dnn):     
        self.df = df_dnn

    def __len__(self):
        return len(self.df)

    def __getitem__(self, idx):
        user = self.df.iloc[idx]['user']
        time = self.df.iloc[idx]['time']
        app = self.df.iloc[idx]['app']
        app_seq = self.df.iloc[idx]['app_seq']
        return (torch.LongTensor([user]), torch.LongTensor([time]), torch.LongTensor([app]), torch.LongTensor(app_seq))

class DNN(nn.Module):
    def __init__(self, n_users, n_times, n_apps, dim, hidden, seq_length):
        super(DNN, self).__init__()
        self.user_emb = nn.Embedding(n_users, dim)
        self.time_emb = nn.Embedding(n_times, dim)
        self.app_emb = nn.Embedding(n_apps, dim)

        self.nn1 = nn.Linear(dim * (seq_length + 2), hidden)
        self.nn2 = nn.Linear(hidden, hidden)
        self.classifier = nn.Linear(hidden, n_apps)
        self.sf = nn.Softmax(dim=1)

    def forward(self, users, times, app_seq):
        # users [batch_size, 1]
        # times [batch_size, 1]
        # app_seq [batch_size, seq_length]

        batch_size = users.size(0)
        user_vector = self.user_emb(users) # [batch_size, 1, dim]
        time_vector = self.time_emb(times) # [batch_size, 1, dim]
        app_seq_vector = self.app_emb(app_seq) # [batch_size, seq_length, dim]

        input_vector = torch.cat([user_vector, time_vector, app_seq_vector], axis=1) # [batch_size, seq_length+2, dim]
        input_vector = input_vector.view(batch_size, -1)

        x = self.nn1(input_vector)
        x = F.relu(x)
        x = self.nn2(x)
        x = F.relu(x)
        return self.classifier(x)