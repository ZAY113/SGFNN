import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import Dataset
import math
import dgl
import dgl.nn as dglnn
from dgl.dataloading import DataLoader as DGLDataLoader, NeighborSampler, EdgeDataLoader, negative_sampler
import dgl.function as fn

class GNNDataset(Dataset):
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
        time_seq = self.df.iloc[idx]['time_seq']
        return (torch.LongTensor([user]), torch.LongTensor([time]), torch.LongTensor([loc]), torch.LongTensor([app]), torch.LongTensor(app_seq), torch.LongTensor(time_seq))


class GNN(nn.Module):
    def __init__(self, n_users, n_times, n_locs, n_apps, hidden_dim, dim, seq_length, graph, device):
        super(GNN, self).__init__()

        #self.user_app_emb = nn.Linear(1, dim)
        self.tla_emb = nn.Embedding(n_times+n_locs+n_apps, dim)
        self.user_emb = nn.Embedding(n_users, dim)
        self.time_emb = nn.Embedding(n_times, dim)
        self.loc_emb = nn.Embedding(n_locs, dim)
        self.app_emb = nn.Embedding(n_apps, dim)
        self.dim = dim
        self.seq_length = seq_length
        self.n_users = n_users
        self.n_times = n_times
        self.n_locs = n_locs
        self.n_apps = n_apps

        self.all_app_vector = torch.nan

        self.input_dim = self.dim * 4
        self.hidden_dim = hidden_dim
        #self.hidden_dim = self.input_dim

        # graph section
        self.graph = graph.to(device)
        self.device = device
        self.layers = nn.ModuleList()
        # three-layer GraphSAGE-mean
        #self.layers.append(dglnn.GATConv(dim, dim, num_heads=8, allow_zero_in_degree=True))
        #self.layers.append(dglnn.SAGEConv(10, 10, 'mean'))
        #self.layers.append(dglnn.SAGEConv(10, 10, 'mean'))
        self.layers.append(dglnn.SAGEConv(dim, dim, 'mean'))
        self.layers.append(dglnn.SAGEConv(dim, dim, 'mean'))
        self.headmerge = nn.Linear(8, 1)
        self.linear = nn.Linear(10, dim)
        

    def graph_layer(self, nodes_idx_uniq):
        sg = dgl.node_subgraph(self.graph, nodes_idx_uniq)
        """
        print("--------sub graph---------")
        print(sg)
        print("--------------------------")
        """
        x = sg.srcdata['feature']

        h = self.tla_emb(x)

        for l, layer in enumerate(self.layers):
            h = layer(sg, h)
            #h = F.group_norm(h, 1)
            h = F.relu(h)

        sg.srcdata['vec'] = h
        return h

    def cosineSim(self, vec1, vec2):
        norm1 = vec1 / vec1.norm(dim=-1, keepdim=True)
        norm2 = vec2 / vec2.norm(dim=-1, keepdim=True)
        return (norm1 * norm2).sum(dim=-1)

    def computeSim(self, h, batch_size):
        h = h.reshape(batch_size, self.seq_length+2, self.dim)
        time_vector = h[:, 0:1, :]
        loc_vector = h[:, 1:2, :]
        app_seq_vector = h[:, 2:, :]

        sim_t = self.cosineSim(time_vector, (loc_vector + app_seq_vector) / 2).sum(-1)
        sim_l = self.cosineSim(loc_vector, (time_vector + app_seq_vector) / 2).sum(-1)
        sim_a = self.cosineSim(app_seq_vector, (time_vector + loc_vector) / 2).sum(-1)
        loss = (sim_t + sim_l + sim_a)
        return loss


    def forward(self, users, times, locs, app_seq):
        # users [batch_size, 1]
        # times [batch_size, 1]
        # app_seq [batch_size, seq_length]

        # nodes_idx: [app, loc, time]
        # 获取所有node
        loc_node_idx = (self.n_apps + locs)
        time_node_idx = (self.n_apps + self.n_locs + times)
        app_nodes_idx = app_seq
        
        # time-loc-app图
        nodes_idx = torch.cat([time_node_idx, loc_node_idx, app_nodes_idx], dim=1)
        
        # print(nodes_idx.shape)
        nodes_idx = nodes_idx.reshape(-1)     
        
        batch_size = users.size(0)
        
        # 负采样
        neg_u = torch.randint(0, self.n_users, (batch_size, 1)).long().to(self.device)
        neg_t = torch.randint(0, self.n_times, (batch_size, 1)).long().to(self.device)
        neg_l = torch.randint(0, self.n_locs, (batch_size, 1)).long().to(self.device)
        neg_a = torch.randint(0, self.n_apps, (batch_size, self.seq_length)).long().to(self.device)
        # 获取所有node
        neg_loc_node_idx = (self.n_apps + neg_l)
        neg_time_node_idx = (self.n_apps + self.n_locs + neg_t)
        neg_app_nodes_idx = neg_a
        # time-loc-app负图
        neg_nodes_idx = torch.cat([neg_time_node_idx, neg_loc_node_idx, neg_app_nodes_idx], dim=1)
        neg_nodes_idx = neg_nodes_idx.reshape(-1)

        batch_size = users.size(0)

        # 正图
        pos_h = self.graph_layer(nodes_idx)
        pos_loss = self.computeSim(pos_h, batch_size)

        # 负图
        neg_h = self.graph_layer(neg_nodes_idx)
        neg_loss = self.computeSim(neg_h, batch_size)

        return pos_loss, neg_loss

    def appEmbedding(self):
        # app nodes [0, num_apps-1]
        all_app_nodes_idx = torch.arange(0, self.n_apps, dtype=int).to(self.device)
        self.all_app_vector = self.graph_layer(all_app_nodes_idx)
        
    def inference(self, times, locs, app_seq, time_diff_seq, args):
        loc_node_idx = (self.n_apps + locs)
        time_node_idx = (self.n_apps + self.n_locs + times)
        app_seq_nodes_idx = app_seq

        
        # time-loc-app图
        nodes_idx = torch.cat([time_node_idx, loc_node_idx, app_seq_nodes_idx], dim=1)
        nodes_idx = nodes_idx.reshape(-1)
        pos_h = self.graph_layer(nodes_idx)

        batch_size = times.size(0)

        h = pos_h.reshape(batch_size, self.seq_length+2, self.dim)
        time_vector = h[:, 0:1, :]
        loc_vector = h[:, 1:2, :]
        app_seq_vector = h[:, 2:self.seq_length+2, :]


        minitues = 7
        time_diff = torch.exp(- time_diff_seq / minitues).unsqueeze(-1)
        loc_vec = loc_vector.repeat(1, self.seq_length, 1)
        loc_sum_vec = (time_diff * loc_vec).sum(dim=1)
        app_sum_vec = (time_diff * app_seq_vector).sum(dim=1)
        user_vec = args.beta * loc_sum_vec + (1 - args.beta) * app_sum_vec
        user_vec = user_vec.unsqueeze(1)
        scores = self.cosineSim(user_vec, self.all_app_vector)
        return scores