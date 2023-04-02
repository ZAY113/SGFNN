import os
import argparse
import random
import ast
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import torch 
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader
from sklearn.model_selection import train_test_split
from GNN import *
import datetime
import time as clock

from sklearn.preprocessing import LabelEncoder

def prep_time(t):
    t = t[:-2]  # 去除分钟
    weekday = datetime.datetime.strptime(t[:-2], '%Y%m%d').weekday()
    if weekday >= 5:
        weekday = '1'
    else:
        weekday = '0'
    return '{}_{}'.format(weekday, t[-2:])  # 取周末/工作日；小时

def computeLoss(pos_score, neg_score):
    # 交叉熵损失
    """
    pos_score = torch.sigmoid(pos_score)
    neg_score = torch.sigmoid(-neg_score)
    loss = (-torch.log(pos_score) - torch.log(neg_score)).sum()
    """
    # 交叉熵损失
    pos_score = torch.sigmoid(pos_score)
    neg_score = torch.sigmoid(-neg_score)
    score_unsuper = torch.cat([pos_score, neg_score])
    pos_label = torch.ones_like(pos_score)
    neg_label = torch.zeros_like(neg_score)
    labels = torch.cat([pos_label, neg_label])
    loss = F.binary_cross_entropy_with_logits(score_unsuper, labels)
    loss = torch.mean(loss)
    
    return loss


def evaluateModel(model, criterion, data_loader, device, args):
    model.eval()
    l_sum, n = 0.0, 0
    with torch.no_grad():
        for user, time, loc, app, app_seq, time_seq in data_loader:
            user = user.to(device)
            time = time.to(device)
            loc = loc.to(device)
            target = app.to(device)
            app_seq = app_seq.to(device)

            pos_score, neg_score = model(user, time, loc, app_seq) # [batch_size, num_apps]
            l = computeLoss(pos_score, neg_score)
            #l = criterion(scores, target.view(-1))
            l_sum += l.item() * target.shape[0]
            n += target.shape[0]
        return l_sum / n


def predictModel(model, data_loader, device, args):
    model.eval()
    Ks = [1, 5, 10]
    corrects = [0, 0, 0]
    model.appEmbedding()
    with torch.no_grad():
        for user, time, loc, app, app_seq, time_seq in data_loader:
            user = user.to(device)
            time = time.to(device)
            loc = loc.to(device)
            target = app.to(device)
            app_seq = app_seq.to(device)
            time_seq = time_seq.to(device)
#===================================================================================================
            #model.appEmbedding()
            scores = model.inference(time, loc, app_seq, time_seq, args)
            for idx, k in enumerate(Ks):
                correct = torch.sum(torch.eq(torch.topk(scores, dim=1, k=k).indices, target)).item()
                corrects[idx] += correct
    return corrects


def trainModel(name, mode, train, val, n_users, n_times, n_locs, n_apps, args, g, device):
    with open('Log/' + KEYWORD + '_' + name + '_log.txt', 'a') as f:
        f.write("epoch: {}\n".format(args.epoch))
        f.write("batch_size: {}\n".format(args.batch_size))
        f.write("dim: {}\n".format(args.dim))
        f.write("seq_length: {}\n".format(args.seq_length))
        f.write("hidden: {}\n".format(args.hidden))
        f.write("lr: {}\n".format(args.lr))
        f.write("seed: {}\n".format(args.seed))
        f.write("trainval_split: {}\n".format(args.trainval_split))
        f.write("patience: {}\n".format(args.patience))
        f.write("model_type: {}\n".format(args.model_type))
        f.write("topk: {}\n".format(args.topk))
        f.write("alpha: {}\n".format(args.alpha))
        f.write("beta: {}\n".format(args.beta))

    train_dataset = GNNDataset(train)   # (636139, 5)
    val_dataset = GNNDataset(val)       # (90877, 5)

    train_loader = DataLoader(train_dataset, batch_size=args.batch_size, shuffle=True)
    val_loader = DataLoader(val_dataset, batch_size=args.batch_size, shuffle=True)
    print('Model Training Started ...', clock.ctime())
    
    
    # model & optimizer
    model = GNN(n_users, n_times, n_locs, n_apps, args.hidden, args.dim, args.seq_length, g, device)
    # ================ 接着训练 ==============
    # model.load_state_dict(torch.load('model/' + KEYWORD + '_' + name + '.pt'))
    # =======================================
    model.to(device)
    optimizer = optim.Adam(model.parameters(), lr=args.lr)
    criterion = nn.CrossEntropyLoss()
    
    # train & val
    min_val_loss = np.inf
    wait = 0
    p_itr = 500
    loss_list = []
    for epoch in range(args.epoch):
        starttime = datetime.datetime.now()
        """
        train_loss = 0
        val_loss = 0
        if epoch != 0:
        """
        loss_sum, n = 0.0, 0
        model.train()
        for i, (user, time, loc, app, app_seq, time_seq) in enumerate(train_loader):
            user = user.to(device)
            time = time.to(device)
            loc = loc.to(device)
            target = app.to(device)
            app_seq = app_seq.to(device)

            optimizer.zero_grad()
            pos_score, neg_score = model(user, time, loc, app_seq) # [batch_size, num_apps]
            # n类: scores表示C类的概率[c1,c2,...,cn]; target表示groundtruth的类下标
            loss = computeLoss(pos_score, neg_score)
            #loss = criterion(scores, target.view(-1))
            loss.backward()
            optimizer.step()
            loss_sum += loss.item() * target.shape[0]
            # 便于观察训练进度
            n += target.shape[0]
            if (i+1) % p_itr == 0:
                print("[TRAIN] Epoch: {} / Iter: {} Loss - {}".format(epoch+1, i+1, loss_sum/n))
        train_loss = loss_sum / n
        val_loss = evaluateModel(model, criterion, val_loader, device, args)
        #val_loss = evaluateModel(model, criterion, val_loader, device, args)
        loss_list.append([train_loss, val_loss])
        if val_loss < min_val_loss:
            wait = 0
            min_val_loss = val_loss
            torch.save(model.state_dict(), 'model/' + KEYWORD + '_' + name + '.pt')
        else:
            wait += 1
            if wait == args.patience:
                print('Early stopping at epoch: %d' % epoch)
                break
        endtime = datetime.datetime.now()
        epoch_time = (endtime - starttime).seconds
        val_corrects = predictModel(model, val_loader, device, args)
        val_accs = [x/len(val) for x in val_corrects]
        print("epoch", epoch, "time used:", epoch_time, "seconds", "train loss:", train_loss, "validation loss:", val_loss)
        with open('Log/' + KEYWORD + '_' + name + '_log.txt', 'a') as f:
            f.write("%s, %d, %s, %d, %s, %s, %.10f, %s, %.10f\n" % 
                ("epoch", epoch, "time used:", epoch_time, "seconds", "train loss:", train_loss, "validation loss:", val_loss))
            f.write("[Train] Val: - Acc: %.5f / %.5f / %.5f\n" % (val_accs[0], val_accs[1], val_accs[2]))
        print("[Train] Val: - Acc: {:.5f} / {:.5f} / {:.5f}".format(val_accs[0], val_accs[1], val_accs[2]))
    #train_loss = evaluateModel(model, criterion, train_loader, device)
    #train_corrects = predictModel(model, train_loader, device)
    #train_accs = [x/len(train) for x in train_corrects]
    #print("%s, %s, Train Loss, %.10f\n" % (name, mode, train_loss))
    #print("[EVALUATION] Train: - Acc: {:.5f} / {:.5f} / {:.5f}".format(train_accs[0], train_accs[1], train_accs[2]))
    val_corrects = predictModel(model, val_loader, device, args)
    val_accs = [x/len(val) for x in val_corrects]
    print("[EVALUATION] Val: - Acc: {:.5f} / {:.5f} / {:.5f}".format(val_accs[0], val_accs[1], val_accs[2]))
    with open('result/' + KEYWORD + '_result.txt', 'a') as f:
        f.write("epoch: {}\n".format(args.epoch))
        f.write("batch_size: {}\n".format(args.batch_size))
        f.write("dim: {}\n".format(args.dim))
        f.write("seq_length: {}\n".format(args.seq_length))
        f.write("hidden: {}\n".format(args.hidden))
        f.write("lr: {}\n".format(args.lr))
        f.write("seed: {}\n".format(args.seed))
        f.write("trainval_split: {}\n".format(args.trainval_split))
        f.write("patience: {}\n".format(args.patience))
        f.write("model_type: {}\n".format(args.model_type))
        f.write("topk: {}\n".format(args.topk))
        f.write("alpha: {}\n".format(args.alpha))
        f.write("beta: {}\n".format(args.beta))
        #f.write("%s, %s, Train Loss, %.10f\n" % (name, mode, train_loss))
        #f.write("[EVALUATION] Train: - Acc: {:.5f} / {:.5f} / {:.5f}".format(train_accs[0], train_accs[1], train_accs[2]))
        f.write("[EVALUATION] Val: - Acc: {:.5f} / {:.5f} / {:.5f}".format(val_accs[0], val_accs[1], val_accs[2]))
    loss_list = np.array(loss_list).transpose()
    np.save('result/' + KEYWORD + '_loss.npy', loss_list)
    print('Model Training Ended ...', clock.ctime())


def testModel(name, mode, test, n_users, n_times, n_locs, n_apps, args, g, device):
    test_dataset = GNNDataset(test)   # (181754, 5)

    test_loader = DataLoader(test_dataset, batch_size=args.batch_size, shuffle=False)
    print('Model Testing Started ...', clock.ctime())
    model = GNN(n_users, n_times, n_locs, n_apps, args.hidden, args.dim, args.seq_length, g, device)
    model.load_state_dict(torch.load('model/' + KEYWORD + '_' + name + '.pt'))
    model.to(device)
    criterion = nn.CrossEntropyLoss()

    #test_loss = evaluateModel(model, criterion, test_loader, device)
    test_corrects = predictModel(model, test_loader, device, args)
    test_accs = [x/len(test) for x in test_corrects]
    #print("%s, %s, Train Loss, %.10f\n" % (name, mode, test_loss))
    print("[EVALUATION] Test: - Acc: {:.5f} / {:.5f} / {:.5f}".format(test_accs[0], test_accs[1], test_accs[2]))
    with open('result/' + KEYWORD + '_result.txt', 'a') as f:
        #print("%s, %s, Train Loss, %.10f\n" % (name, mode, test_loss))
        f.write("[EVALUATION] Test: - Acc: {:.5f} / {:.5f} / {:.5f}".format(test_accs[0], test_accs[1], test_accs[2]))
    print('Model Testing Ended ...', clock.ctime())

def weight_norm(df):
    min_value = df['weight'].min()
    max_value = df['weight'].max()
    df['weight_norm'] = df['weight'].apply(lambda x: (x-min_value)/(max_value-min_value))
    return df

def parse_args():
    parser = argparse.ArgumentParser(description="Pytorch implementation of AppUsage2Vec model")

    parser.add_argument('--epoch', type=int, default=30, help="The number of epochs")
    parser.add_argument('--batch_size', type=int, default=32, help="The size of batch")
    parser.add_argument('--dim', type=int, default=100, help="The embedding size of users and apps")
    parser.add_argument('--seq_length', type=int, default=4, help="The length of previously used app sequence")
    parser.add_argument('--hidden', type=int, default=200, help="The hidden dim in DNN")
    parser.add_argument('--lr', type=float, default=0.01, help="Learning rate for optimizer")
    parser.add_argument('--seed', type=int, default=2021, help="Random seed")
    parser.add_argument('--trainval_split', type=float, default=0.125, help="train:0.8, val:0.8*trainval_split")
    parser.add_argument('--patience', type=int, default=10, help="early stop")
    parser.add_argument('--model_type', type=str, default='GRU', help="model type: tf/LSTM/GRU")
    parser.add_argument('--alpha', type=float, default=3, help="Discount oefficient for loss function")
    parser.add_argument('--topk', type=float, default=1, help="Topk for loss function")
    parser.add_argument('--beta', type=float, default=0.5, help="the section of user vec(location and app)")
    return parser.parse_args()


MODELNAME = "GNN"
KEYWORD = 'pred_Public' + '_' + MODELNAME + '_' + datetime.datetime.now().strftime("%y%m%d%H%M")
#KEYWORD = 'pred_Public_GNN_2302210737'

def main():
    args = parse_args()

    # random seed
    random.seed(args.seed)
    torch.manual_seed(args.seed)

    # device 
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print("Device: {}".format(device))
    if torch.cuda.device_count() > 0:
        torch.cuda.manual_seed_all(args.seed)

    df_usage = pd.read_csv('../data/baseline_loc_time.txt', sep='\t')
    df_usage['app_seq'] = df_usage['app_seq'].apply(ast.literal_eval)
    df_usage['time_seq'] = df_usage['time_seq'].apply(ast.literal_eval)
    df_usage['time'] = df_usage['time'].apply(lambda x: str(x))
    # time的转换 [示例：0_13，周末为1，工作日为0；13表示13点]
    df_usage['time'] = df_usage['time'].apply(lambda x: prep_time(x))

    # encoder
    user_encoder = LabelEncoder()
    time_encoder = LabelEncoder()
    app_encoder = LabelEncoder()
    loc_encoder = LabelEncoder()

    # 特征编码
    user_encoder.fit(df_usage['user'].unique())
    time_encoder.fit(df_usage['time'].unique())
    all_apps = list(df_usage['app'].unique())
    for app_seq in df_usage['app_seq']:
        all_apps.extend(app_seq)
    app_encoder.fit(list(set(all_apps)))
    loc_encoder.fit(df_usage['location'].unique())

    stratify_seed = df_usage['user']
    df_usage['user'] = user_encoder.transform(df_usage['user'])
    df_usage['time'] = time_encoder.transform(df_usage['time'])
    df_usage['app'] = app_encoder.transform(df_usage['app'])
    df_usage['app_seq'] = df_usage['app_seq'].apply(lambda x: app_encoder.transform(x))
    df_usage['location'] = loc_encoder.transform(df_usage['location'])
    
    # 输入特征向量的维度
    num_users = len(df_usage['user'].unique())
    num_times = len(df_usage['time'].unique())
    num_apps = len(app_encoder.classes_)
    num_locs = len(df_usage['location'].unique())

    # split: trian/val/test
    trainval, test = train_test_split(df_usage, test_size=0.2, random_state=2021, stratify=df_usage['user'])
    train, val = train_test_split(trainval, test_size=args.trainval_split, random_state=2021, stratify=trainval['user'])    


    # 构建time-location-app异质图
    edge_list = df_usage[['time', 'location', 'app']]

    time_node = edge_list['time']
    loc_node = edge_list['location']
    app_node = edge_list['app']

    hetero_graph = dgl.heterograph({
        ('time', 'use', 'app'): (time_node, app_node),
        ('app', 'used-by', 'time'): (app_node, time_node),
        ('time', 'visit', 'location'): (time_node, loc_node),
        ('location', 'visited-by', 'time'): (loc_node, time_node),
        ('location', 'open', 'app'): (loc_node, app_node),
        ('app', 'opened-by', 'location'): (app_node, loc_node)
    })
    time_node_feat = hetero_graph.nodes('time')
    loc_node_feat = hetero_graph.nodes('location')
    app_node_feat = hetero_graph.nodes('app')
    hetero_graph.nodes['time'].data['feature'] = time_node_feat
    hetero_graph.nodes['location'].data['feature'] = loc_node_feat
    hetero_graph.nodes['app'].data['feature'] = app_node_feat

    # 异质图转简单图
    g = dgl.to_homogeneous(hetero_graph, ndata=['feature'])
    g = dgl.add_self_loop(g)

    print(g)
    print(KEYWORD, 'training started', clock.ctime())
    trainModel(MODELNAME, 'train', train, val, num_users, num_times, num_locs, num_apps, args, g, device)
    print(KEYWORD, 'testing started', clock.ctime())
    testModel(MODELNAME, 'test', test, num_users, num_times, num_locs, num_apps, args, g, device)

if __name__ == "__main__":
    main()