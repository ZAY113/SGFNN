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
from AppUsage2Vec_Model import *
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

def evaluateModel(model, criterion, data_loader, device, args):
    model.eval()
    l_sum, n = 0.0, 0
    with torch.no_grad():
        for i, (data, targets) in enumerate(data_loader):
            users = data[0].to(device)
            time_vecs = data[1].to(device)
            app_seqs = data[2].to(device)
            time_seqs = data[3].to(device)
            targets = targets.to(device)

            scores = model(users, time_vecs, app_seqs, time_seqs) # [batch_size, num_apps]
            # n类: scores表示C类的概率[c1,c2,...,cn]; target表示groundtruth的类下标
            #l = computeLoss(scores, targets, device, args)
            l = criterion(scores, targets.view(-1))
            l_sum += l.item() * targets.shape[0]
            n += targets.shape[0]
        return l_sum / n

def predictModel(model, data_loader, device):
    model.eval()
    Ks = [1, 5, 10]
    corrects = [0, 0, 0]
    with torch.no_grad():
        for i, (data, targets) in enumerate(data_loader):
            users = data[0].to(device)
            time_vecs = data[1].to(device)
            app_seqs = data[2].to(device)
            time_seqs = data[3].to(device)
            targets = targets.to(device)

            scores = model(users, time_vecs, app_seqs, time_seqs) # [batch_size, num_apps]
            for idx, k in enumerate(Ks):
                correct = torch.sum(torch.eq(torch.topk(scores, dim=1, k=k).indices, targets)).item()
                corrects[idx] += correct
    return corrects

def computeLoss(scores, targets, device, args):
    preds = torch.topk(scores, dim=1, k=args.topk).indices     # [batch_size, k]
    indicator = torch.sum(torch.eq(preds, targets), dim=1)  # [batch_size]
    coefficient = torch.pow(torch.Tensor([args.alpha] * indicator.size(0)).to(device), indicator)  # [batch_size]

    loss = F.cross_entropy(scores, targets.view(-1), reduction='none')
    loss = torch.mean(torch.mul(coefficient, loss))
    return loss

def trainModel(name, mode, train, val, n_users, n_apps, args, device):
    with open('Log/' + KEYWORD + '_' + name + '_log.txt', 'a') as f:
        f.write("epoch: {}\n".format(args.epoch))
        f.write("batch_size: {}\n".format(args.batch_size))
        f.write("dim: {}\n".format(args.dim))
        f.write("seq_length: {}\n".format(args.seq_length))
        f.write("num_layers: {}\n".format(args.num_layers))
        f.write("alpha: {}\n".format(args.alpha))
        f.write("topk: {}\n".format(args.topk))
        f.write("lr: {}\n".format(args.lr))
        f.write("seed: {}\n".format(args.seed))
        f.write("trainval_split: {}\n".format(args.trainval_split))
        f.write("patience: {}\n".format(args.patience))

    train_dataset = AppUsage2VecDataset(train)   # (636139, 5)
    val_dataset = AppUsage2VecDataset(val)       # (90877, 5)

    train_loader = DataLoader(train_dataset, batch_size=args.batch_size, shuffle=True)
    val_loader = DataLoader(val_dataset, batch_size=args.batch_size, shuffle=True)
    print('Model Training Started ...', clock.ctime())
    
    # model & optimizer
    model = AppUsage2Vec(n_users, n_apps, args.dim, args.seq_length, args.num_layers, args.alpha, args.topk)
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
        loss_sum, n = 0.0, 0
        model.train()
        for i, (data, targets) in enumerate(train_loader):
            users = data[0].to(device)
            time_vecs = data[1].to(device)
            app_seqs = data[2].to(device)
            time_seqs = data[3].to(device)
            targets = targets.to(device)

            optimizer.zero_grad()
            scores = model(users, time_vecs, app_seqs, time_seqs) # [batch_size, num_apps]
            # n类: scores表示C类的概率[c1,c2,...,cn]; target表示groundtruth的类下标
            #loss = computeLoss(scores, targets, device, args)
            loss = criterion(scores, targets.view(-1))
            loss.backward()
            optimizer.step()
            loss_sum += loss.item() * targets.shape[0]
            # 便于观察训练进度
            n += targets.shape[0]
            if (i+1) % p_itr == 0:
                print("[TRAIN] Epoch: {} / Iter: {} Loss - {}".format(epoch+1, i+1, loss_sum/n))
        train_loss = loss_sum / n
        val_loss = evaluateModel(model, criterion, val_loader, device, args)
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
        print("epoch", epoch, "time used:", epoch_time, "seconds", "train loss:", train_loss, "validation loss:", val_loss)
        with open('Log/' + KEYWORD + '_' + name + '_log.txt', 'a') as f:
            f.write("%s, %d, %s, %d, %s, %s, %.10f, %s, %.10f\n" % 
                ("epoch", epoch, "time used:", epoch_time, "seconds", "train loss:", train_loss, "validation loss:", val_loss))
    
    #train_loss = evaluateModel(model, criterion, train_loader, device)
    #train_corrects = predictModel(model, train_loader, device)
    #train_accs = [x/len(train) for x in train_corrects]
    #print("%s, %s, Train Loss, %.10f\n" % (name, mode, train_loss))
    #print("[EVALUATION] Train: - Acc: {:.5f} / {:.5f} / {:.5f}".format(train_accs[0], train_accs[1], train_accs[2]))
    val_corrects = predictModel(model, val_loader, device)
    val_accs = [x/len(val) for x in val_corrects]
    print("[EVALUATION] Val: - Acc: {:.5f} / {:.5f} / {:.5f}".format(val_accs[0], val_accs[1], val_accs[2]))
    with open('result/' + KEYWORD + '_result.txt', 'a') as f:
        f.write("epoch: {}\n".format(args.epoch))
        f.write("batch_size: {}\n".format(args.batch_size))
        f.write("dim: {}\n".format(args.dim))
        f.write("seq_length: {}\n".format(args.seq_length))
        f.write("num_layers: {}\n".format(args.num_layers))
        f.write("alpha: {}\n".format(args.alpha))
        f.write("topk: {}\n".format(args.topk))
        f.write("lr: {}\n".format(args.lr))
        f.write("seed: {}\n".format(args.seed))
        f.write("trainval_split: {}\n".format(args.trainval_split))
        f.write("patience: {}\n".format(args.patience))
        #f.write("%s, %s, Train Loss, %.10f\n" % (name, mode, train_loss))
        #f.write("[EVALUATION] Train: - Acc: {:.5f} / {:.5f} / {:.5f}".format(train_accs[0], train_accs[1], train_accs[2]))
        f.write("[EVALUATION] Val: - Acc: {:.5f} / {:.5f} / {:.5f}".format(val_accs[0], val_accs[1], val_accs[2]))
    loss_list = np.array(loss_list).transpose()
    np.save('result/' + KEYWORD + '_loss.npy', loss_list)
    print('Model Training Ended ...', clock.ctime())

def testModel(name, mode, test, n_users, n_apps, args, device):
    test_dataset = AppUsage2VecDataset(test)   # (181754, 5)

    test_loader = DataLoader(test_dataset, batch_size=args.batch_size, shuffle=False)
    print('Model Testing Started ...', clock.ctime())
    model = AppUsage2Vec(n_users, n_apps, args.dim, args.seq_length, args.num_layers, args.alpha, args.topk)
    model.load_state_dict(torch.load('model/' + KEYWORD + '_' + name + '.pt'))
    model.to(device)
    #criterion = nn.CrossEntropyLoss()
    #test_loss = evaluateModel(model, criterion, test_loader, device)
    test_corrects = predictModel(model, test_loader, device)
    test_accs = [x/len(test) for x in test_corrects]
    #print("%s, %s, Train Loss, %.10f\n" % (name, mode, test_loss))
    print("[EVALUATION] Test: - Acc: {:.5f} / {:.5f} / {:.5f}".format(test_accs[0], test_accs[1], test_accs[2]))
    with open('result/' + KEYWORD + '_result.txt', 'a') as f:
        #print("%s, %s, Train Loss, %.10f\n" % (name, mode, test_loss))
        f.write("[EVALUATION] Test: - Acc: {:.5f} / {:.5f} / {:.5f}".format(test_accs[0], test_accs[1], test_accs[2]))
    print('Model Testing Ended ...', clock.ctime())

def parse_args():
    parser = argparse.ArgumentParser(description="Pytorch implementation of AppUsage2Vec model")

    parser.add_argument('--epoch', type=int, default=100, help="The number of epochs")
    parser.add_argument('--batch_size', type=int, default=256, help="The size of batch")
    parser.add_argument('--dim', type=int, default=200, help="The embedding size of users and apps")
    parser.add_argument('--seq_length', type=int, default=4, help="The length of previously used app sequence")
    parser.add_argument('--num_layers', type=int, default=2, help="Number of layers in DNN")
    parser.add_argument('--alpha', type=float, default=3, help="Discount oefficient for loss function")
    parser.add_argument('--topk', type=float, default=1, help="Topk for loss function")
    parser.add_argument('--lr', type=float, default=0.0005, help="Learning rate for optimizer")
    parser.add_argument('--seed', type=int, default=2021, help="Random seed")
    parser.add_argument('--trainval_split', type=float, default=0.125, help="train:0.8, val:0.8*trainval_split")
    parser.add_argument('--patience', type=int, default=10, help="early stop")

    return parser.parse_args()


MODELNAME = "AppUsage2Vec"
#KEYWORD = 'pred_Public' + '_' + MODELNAME + '_' + datetime.datetime.now().strftime("%y%m%d%H%M")
KEYWORD = 'pred_Public_AppUsage2Vec_2301061606'
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

    trainval = pd.read_csv('data/train.txt', sep='\t')
    test = pd.read_csv('data/test.txt', sep='\t')
    train, val = train_test_split(trainval, test_size=args.trainval_split, random_state=2021, stratify=trainval['user'])

    #print(train.shape)
    #print(val.shape)
    #print(test.shape)

    num_users = len(open('data/user2id.txt', 'r').readlines())
    num_apps = len(open('data/app2id.txt', 'r').readlines())

    #print(KEYWORD, 'training started', clock.ctime())
    #trainModel(MODELNAME, 'train', train, val, num_users, num_apps, args, device)
    print(KEYWORD, 'testing started', clock.ctime())
    testModel(MODELNAME, 'test', test, num_users, num_apps, args, device)

if __name__ == "__main__":
    main()