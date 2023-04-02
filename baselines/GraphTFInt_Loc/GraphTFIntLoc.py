import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import Dataset
import math
import dgl
import dgl.nn as dglnn
from dgl.dataloading import DataLoader as DGLDataLoader, NeighborSampler, EdgeDataLoader, negative_sampler
import dgl.function as fn

class GraphTFIntLocDataset(Dataset):
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

class DotProductPredictor(nn.Module):
    def forward(self, graph, h):
        with graph.local_scope():
            graph.ndata['h'] = h
            graph.apply_edges(fn.u_dot_v('h', 'h', 'score'))
            return graph.edata['score']

class GraphTFIntLoc(nn.Module):
    def __init__(self, n_users, n_times, n_locs, n_apps, hidden_dim, dim, seq_length, graph, device):
        super(GraphTFIntLoc, self).__init__()

        #self.user_app_emb = nn.Linear(1, dim)
        self.user_app_emb = nn.Embedding(n_users+n_apps, dim)
        self.time_app_emb = nn.Embedding(n_times+n_apps, dim)
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

        self.input_dim = self.dim * 4
        self.hidden_dim = hidden_dim
        #self.hidden_dim = self.input_dim
        self.input_layer = nn.Linear(self.input_dim, self.hidden_dim)
        self.encoder_layer = nn.TransformerEncoderLayer(d_model=self.hidden_dim, nhead=8, batch_first=True)
        #self.encoder_layer = nn.TransformerEncoderLayer(d_model=self.hidden_dim, nhead=8, dim_feedforward=self.hidden_dim*3, batch_first=True)
        self.transformer_encoder = nn.TransformerEncoder(self.encoder_layer, num_layers=1)
        self.time_encoder_layer = nn.TransformerEncoderLayer(d_model=self.seq_length, nhead=1, dim_feedforward=self.seq_length, batch_first=True)
        self.time_encoder = nn.TransformerEncoder(self.time_encoder_layer, num_layers=1)
        self.time_linear = nn.Linear(self.seq_length, 1)
        self.classifier = nn.Linear(self.hidden_dim, n_apps)

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
        
        self.pred = DotProductPredictor()

    def construct_negative_graph(self, graph, k):
        src, dst = graph.edges()

        neg_src = src.repeat_interleave(k)
        neg_dst = torch.randint(0, graph.num_nodes(), (len(src) * k,)).to(self.device)
        return dgl.graph((neg_src, neg_dst), num_nodes=graph.num_nodes(), device=self.device)


    def graph_layer(self, nodes_idx_uniq):
        sg = dgl.node_subgraph(self.graph, nodes_idx_uniq)
        """
        print("--------sub graph---------")
        print(sg)
        print("--------------------------")
        """
        x = sg.srcdata['feature']

        h = self.user_app_emb(x)
        #h = self.time_app_emb(x)

        for l, layer in enumerate(self.layers):
            h = layer(sg, h)
            #h = F.group_norm(h, 1)
            h = F.relu(h)

        neg_sg = self.construct_negative_graph(sg, k=5)
        pos_score = self.pred(sg, h)
        neg_score = self.pred(neg_sg, h)

        return h, pos_score, neg_score
        
    def cosineSim(self, vec1, vec2):
        norm1 = vec1 / vec1.norm(dim=-1, keepdim=True)
        norm2 = vec2 / vec2.norm(dim=-1, keepdim=True)
        return (norm1 * norm2).sum(dim=-1)

    def hyperEageLoss(self, u_vec, t_vec, l_vec, a_seq_vec):
        u_vec = u_vec / u_vec.norm(dim=-1, keepdim=True)
        t_vec = t_vec / t_vec.norm(dim=-1, keepdim=True)
        l_vec = l_vec / l_vec.norm(dim=-1, keepdim=True)
        a_seq_vec = a_seq_vec / a_seq_vec.norm(dim=-1, keepdim=True)

        best_vec = (u_vec + t_vec + l_vec + a_seq_vec) / 4

        hyper_score = self.cosineSim(u_vec, best_vec) + self.cosineSim(t_vec, best_vec) + self.cosineSim(l_vec, best_vec) + self.cosineSim(a_seq_vec, best_vec)
        return hyper_score.sum() / 4


    def forward(self, users, times, locs, app_seq):
        # users [batch_size, 1]
        # times [batch_size, 1]
        # app_seq [batch_size, seq_length]


        # 获取所有node
        user_node_idx = (users + self.n_apps)
        time_node_idx = (times + self.n_apps)
        app_nodes_idx = app_seq
        # print(user_node_idx.shape)
        # print(app_nodes_idx.shape)
        
        # user-app图
        nodes_idx = torch.cat([user_node_idx, app_nodes_idx], dim=1)
        # time-app图
        #nodes_idx = torch.cat([time_node_idx, app_nodes_idx], dim=1)
        
        # print(nodes_idx.shape)
        nodes_idx = nodes_idx.reshape(-1)
        #node_nums = nodes_idx.shape[0]  # 原始节点数目
        # print(nodes_idx.shape)
        # nodes_idx: shape (256*5,)
        #nodes_idx_uniq = nodes_idx.unique()

        h_g, pos_score, neg_score = self.graph_layer(nodes_idx)
        #h_g, pos_score, neg_score = self.graph_layer(nodes_idx_uniq)
        
        #node_hid = h_g.shape[0]
        #h_g = h_g.permute(1, 0)
        #m = nn.Linear(node_hid, node_nums, device=self.device)
        #h_g = m(h_g)
        #h_g = h_g.permute(1, 0)


        h = self.user_app_emb(nodes_idx)
        #h1 = self.time_app_emb(nodes_idx)

        batch_size = users.size(0)
        h = h.reshape(batch_size, self.seq_length+1, self.dim)

        user_vector = h[:, 0:1, :]
        #time_vector = h[:, 0:1, :]
        app_seq_vector = h[:, 1:, :]
        #print(user_vector.shape)
        #print(app_seq_vector.shape)
        
        #user_vector = self.user_emb(users)  # [batch_size, 1, dim]
        time_vector = self.time_emb(times)  # [batch_size, 1, dim]
        loc_vector = self.loc_emb(locs)  # [batch_size, 1, dim]
        #app_seq_vector = self.app_emb(app_seq)  # [batch_size, seq_length, dim]

        # [batch_size, seq_length, input_dim]
        #--- input_vector = torch.cat([user_vector.repeat(1, self.seq_length, 1), time_vector.repeat(1, self.seq_length, 1), loc_vector.repeat(1, self.seq_length, 1), app_seq_vector], axis=2)

        # graph部分
        h_g = h_g.reshape(batch_size, self.seq_length+1, self.dim)
        user_vector_g = h_g[:, 0:1, :]
        app_seq_vector_g = h_g[:, 1:, :]
        #print("------------save hg-------------")
        #torch.save(h_g.to(torch.device('cpu')), 'case/bip.pth')
        # [batch_size, seq_length, input_dim]
        input_vector_g = torch.cat([user_vector_g.repeat(1, self.seq_length, 1), time_vector.repeat(1, self.seq_length, 1), loc_vector.repeat(1, self.seq_length, 1), app_seq_vector_g], axis=2)

        
        #--- x1 = self.input_layer(input_vector)
        xg = self.input_layer(input_vector_g)

        
        #--- x1 = self.transformer_encoder(x1)
        """
        _, attn = self.encoder_layer.self_attn(xg, xg, xg)
        print("Attn: ")
        print(attn.shape)
        print("------------save attn-------------")
        torch.save(attn.to(torch.device('cpu')), 'case/attn.pth')
        """
        x1 = self.transformer_encoder(xg)

        x2 = x1.permute(0, 2, 1)
        x2 = self.time_encoder(x2)
        x2 = x2.permute(0, 2, 1)
        x1 = x1 + x2
        
        
        # 残差，加强graph sage的影响
        x = x1 + xg
        
        # 只考虑graph sage
        #x = xg

        x = x.permute(0, 2, 1)
        # [B, T, H] -> [B, H, T] -> [B, H, 1]
        x = self.time_linear(x)
        x = x.squeeze(2)


        # 负采样
        neg_u = torch.randint(0, self.n_users, (batch_size, 1)).long().to(self.device)
        neg_t = torch.randint(0, self.n_times, (batch_size, 1)).long().to(self.device)
        neg_l = torch.randint(0, self.n_locs, (batch_size, 1)).long().to(self.device)
        neg_a = torch.randint(0, self.n_apps, (batch_size, self.seq_length)).long().to(self.device)
        # 获取所有node
        neg_user_node_idx = (neg_u + self.n_apps)
        neg_app_nodes_idx = neg_a

        neg_nodes_idx = torch.cat([neg_user_node_idx, neg_app_nodes_idx], dim=1)
        neg_h = self.user_app_emb(neg_nodes_idx)
        neg_user_vector = neg_h[:, 0:1, :]
        neg_app_seq_vector = neg_h[:, 1:, :]
        neg_time_vector = self.time_emb(neg_t)  # [batch_size, 1, dim]
        neg_loc_vector = self.loc_emb(neg_l)  # [batch_size, 1, dim]

        pos_hyper_score = self.hyperEageLoss(user_vector, time_vector, loc_vector, app_seq_vector)
        neg_hyper_score = self.hyperEageLoss(neg_user_vector, neg_time_vector, neg_loc_vector, neg_app_seq_vector)

        """
        print("------------save hypereage-------------")
        torch.save(user_vector.to(torch.device('cpu')), 'case/user_p.pth')
        torch.save(time_vector.to(torch.device('cpu')), 'case/time_p.pth')
        torch.save(loc_vector.to(torch.device('cpu')), 'case/loc_p.pth')
        torch.save(app_seq_vector.to(torch.device('cpu')), 'case/app_p.pth')
        torch.save(neg_user_vector.to(torch.device('cpu')), 'case/user_n.pth')
        torch.save(neg_time_vector.to(torch.device('cpu')), 'case/time_n.pth')
        torch.save(neg_loc_vector.to(torch.device('cpu')), 'case/loc_n.pth')
        torch.save(neg_app_seq_vector.to(torch.device('cpu')), 'case/app_n.pth')
        """
        #x = hidden_last.squeeze(0)
        return self.classifier(x), pos_score, neg_score, pos_hyper_score, neg_hyper_score