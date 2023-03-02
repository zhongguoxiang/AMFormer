import random
import numpy as np
from torch.optim import Adam,Adamax,SGD,NAdam
import torch
from sklearn import metrics
import math
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader
import scipy.io as scio
from sklearn import preprocessing
import matplotlib.pyplot as plt
from sklearn.metrics import accuracy_score
from spot import SPOT
import time

class MyDataset(Dataset):
    def __init__(self, data, label):
        self.data = data
        self.label = label

    def __getitem__(self, index):
        return (torch.as_tensor(self.data[index], dtype=torch.float), torch.as_tensor(self.label[index], dtype=torch.long))

    def __len__(self):
        return len(self.data)

def segment(x,windows_size,step_size):
    x=torch.as_tensor(x)
    output=[]
    for i in range(windows_size,x.shape[0],step_size):
        if i==windows_size:
            output.append(x[0:windows_size,:])
        else:
            output.append(x[i-windows_size:i,:])
    return torch.stack(output)

class featureformer(nn.Module):
    def __init__(self,window_size,input_dim,head):
        super(featureformer, self).__init__()
        self.attention=nn.MultiheadAttention(embed_dim=input_dim,num_heads=head,dropout=0.5,batch_first=True)
        self.normalize1=nn.LayerNorm([window_size,input_dim])
        self.dropout1=nn.Dropout(0.5)
        self.linear=nn.Linear(input_dim,input_dim)
        self.normalize2=nn.LayerNorm([window_size, input_dim])
        self.dropout2=nn.Dropout(0.5)
    def forward(self,x):
        x_output,_= self.attention(query=x, key=x, value=x)
        x_normalize=self.normalize1(x+self.dropout1(x_output))
        x_linear=self.linear(x_normalize)
        return self.normalize2(x_normalize+self.dropout2(x_linear))

class decoderformer(nn.Module):
    def __init__(self,step_size,input_dim,head):
        super(decoderformer, self).__init__()
        self.attention1=nn.MultiheadAttention(embed_dim=input_dim,num_heads=head,dropout=0.5,batch_first=True)
        self.normalize1=nn.LayerNorm([step_size,input_dim])
        self.dropout1=nn.Dropout(0.5)
        self.linear=nn.Linear(input_dim,input_dim)
        self.normalize2=nn.LayerNorm([step_size, input_dim])
        self.dropout2=nn.Dropout(0.5)
        self.attention2=nn.MultiheadAttention(embed_dim=input_dim,num_heads=head,dropout=0.5,batch_first=True)
        self.normalize3=nn.LayerNorm([step_size, input_dim])
        self.dropout3=nn.Dropout(0.5)
    def forward(self,x,y):
        x_output,_=self.attention1(query=x,key=x,value=x)
        x_normalize=self.normalize1(x+self.dropout1(x_output))
        x_output1,_=self.attention2(query=x_normalize,key=y,value=y)
        x_normalize1=self.normalize2(x_normalize+self.dropout2(x_output1))
        x_linear=self.linear(x_normalize1)
        return self.normalize3(x_normalize1+self.dropout3(x_linear))

class liner_ae(nn.Module):
    def __init__(self,num_blocks_encoder,num_blocks_decoder,sample_dim,hidden_dim,window_size,step_size,head):
        super(liner_ae, self).__init__()
        self.linear1=nn.Linear(sample_dim,hidden_dim)
        self.normalize=nn.LayerNorm([step_size,hidden_dim])
        self.linear2=nn.Linear(hidden_dim,sample_dim)
        self.relu=nn.ReLU()
        self.normalize1=nn.LayerNorm([step_size,sample_dim])
        self.normalize2 = nn.LayerNorm([step_size, sample_dim])
        self.attention1 = nn.MultiheadAttention(embed_dim=sample_dim,num_heads=head,batch_first=True)
        self.attention2 = nn.MultiheadAttention(embed_dim=sample_dim, num_heads=head,batch_first=True)
        self.trans=nn.TransformerEncoder(nn.TransformerEncoderLayer(d_model=sample_dim, nhead=head, batch_first=True), num_layers=10)
        self.linear3=nn.Linear(sample_dim,sample_dim)
        self.linear4=nn.Linear(sample_dim,sample_dim)
        self.transformer = nn.Transformer(d_model=sample_dim, batch_first=True, nhead=head, num_encoder_layers=15,
                                          num_decoder_layers=15)
        self.blks_feature=nn.Sequential()
        # self.pos_encoder = PositionalEncoding(sample_dim, 0.1, window_size)
        for i in range(num_blocks_encoder):
            self.blks_feature.add_module("blockfeature"+str(i),featureformer(window_size,sample_dim,head))
        self.blks_decoder=nn.Sequential()
        for j in range(num_blocks_decoder):
            self.blks_decoder.add_module("blockdecoder"+str(j),decoderformer(step_size,sample_dim,head))
    def forward(self,x,x_test):
        x_root=self.relu(self.linear2(self.relu(self.normalize(self.linear1(x_test)))))
        x_encoder=x_test*(x_root/(x_root+0.000001))
        for i,blks in enumerate(self.blks_feature):
            x=blks(x)
        for j,blk in enumerate(self.blks_decoder):
            x_encoder=blk(x_encoder,x)
        return x_encoder,x_root

class metric_loss(torch.nn.Module):
    def __init__(self):
        super(metric_loss, self).__init__()
        self.mse=nn.MSELoss(reduction='none')
    def forward(self,x_rebuild,input_x,parameter):
       distance1 = torch.sum(self.mse(x_rebuild,input_x),dim=2)
       # print(x_rebuild)
       loss_weight=distance1
       weight=torch.ones_like(loss_weight)
       weight1 = torch.pow(weight * parameter, torch.floor(torch.clamp(loss_weight- torch.mean(loss_weight), min=0) / (torch.std(
               loss_weight)+0.000001)))
       distance=torch.mean(distance1*weight1)
       return distance

def mom_threshold(threshold_init):
    threshold_init = threshold_init.cpu().detach().numpy()
    q=0.001
    t=np.percentile(threshold_init,100*(1-0.02))
    peak=threshold_init[np.where(threshold_init>=t)]
    nt=peak.shape[0]
    n=np.sum(threshold_init>=0)
    aver=np.mean(peak-t)
    var=np.var(peak-t)*nt/(nt-1)
    sigma=0.5*aver*(1+aver*aver/var)
    gamma=0.5*(1-aver*aver/var)
    thre=(t+ sigma / gamma * (np.power(q * n / nt, -gamma) - 1))*1
    return thre

def evalution_roc(thresholds_setting,threshold,test_y):
    precision=np.zeros_like(thresholds_setting)
    recall=np.zeros_like(thresholds_setting)
    acc = np.zeros_like(thresholds_setting)
    fpr = np.zeros_like(thresholds_setting)
    for k,i in enumerate(thresholds_setting):
        predict = torch.zeros_like(threshold,dtype=torch.int64).cuda()
        predict[torch.where(threshold>=i)]=1
        temp=torch.sum(test_y,dim=1)*torch.sum(predict,dim=1)
        predict[torch.where(temp>0)[0],:]=test_y[torch.where(temp>0)[0],:]
        predict=predict.view(-1)
        precision[k]=metrics.precision_score(y_true=test_y.view(-1).cpu().detach().numpy(),y_pred=predict.cpu().detach().numpy(),pos_label=1)
        recall[k]=metrics.recall_score(y_true=test_y.view(-1).cpu().detach().numpy(),y_pred=predict.cpu().detach().numpy(),pos_label=1)
        acc[k]=metrics.accuracy_score(y_true=test_y.view(-1).cpu().detach().numpy(),y_pred=predict.cpu().detach().numpy())
        tn, fp, fn, tp =metrics.confusion_matrix(y_true=test_y.view(-1).cpu().detach().numpy(),y_pred=predict.cpu().detach().numpy(),labels=[1,0]).ravel()
        print(tn, fp, fn, tp)
        fpr[k]=fn / (fn + tp)
    return precision,recall,acc,fpr



if __name__=="__main__":
    # smd
    train_x = scio.loadmat('smd_machine1_1_train')['machine1']
    test_x = scio.loadmat('smd_machine1_1_test')['machine1']
    test_y = scio.loadmat('smd_machine1_1_label')['machine1_label']
    num_head = 2

    # psm
    # train_x = scio.loadmat('psm_train')['psm_train']
    # test_x = scio.loadmat('psm_test')['psm_test']
    # test_y = scio.loadmat('psm_test_label')['psm_test_label']
    # num_head = 5

    # msl
    # train_x = np.load('MSL_train.npy')
    # test_x = np.load('MSL_test.npy')
    # test_y = np.load('MSL_test_label.npy')[:,np.newaxis]
    # num_head=5

    # ASD
    # train_x = np.load('asd_train.npy')
    # test_x = np.load('asd_test.npy')
    # test_y = np.load('asd_test_label.npy')
    # num_head=2

    window_size=100
    step_size=50
    # dim=train_x.shape[1]
    train_x_windows = segment(train_x,window_size,step_size)
    test_x_windows = segment(test_x, window_size,step_size)
    test_y_windows = segment(test_y, window_size,step_size)



    device = torch.device('cuda:0')
    batch_size = 128
    num_blocks_encoder = 15
    num_blocks_decoder = 15
    model = liner_ae(num_blocks_encoder,num_blocks_decoder,sample_dim=train_x.shape[1],hidden_dim=2048,window_size=window_size,step_size=step_size,head=num_head)
    model.to(device)


    # training dataset
    dataset = MyDataset(train_x_windows,train_x_windows)
    traindata_loader = DataLoader(dataset, batch_size, shuffle=True,drop_last=True)
    # testing dataset
    dataset = MyDataset(test_x_windows, test_y_windows)
    testdata_loader = DataLoader(dataset, batch_size,drop_last=True)


    # training
    optimizer1 = Adam(model.parameters(), lr=0.001)
    loss0 = metric_loss()
    all_loss1 = []
    model.train()
    for epoch in range(100):
        for batch_idx, batch in enumerate(traindata_loader):
            input_x, _ = tuple(t.to(device) for t in batch)
            x_rebuild,x_root = model(input_x,input_x[:,window_size-step_size:window_size,:])
            loss1= loss0(x_rebuild,input_x[:,window_size-step_size:window_size,:],parameter=0.1)
            loss1.backward()
            optimizer1.step()
            all_loss1.append(loss1.item())

    model.eval()
    threshold1=[]
    threshold2=[]
    test_y=[]
    with torch.no_grad():
        for batch_idx,batch in enumerate(testdata_loader):
            input_x_test,input_y_test=tuple(t.to(device) for t in batch)
            test_rebuild,test_root =model(input_x_test,input_x_test[:,window_size-step_size:window_size,:])
            score1=torch.pow(torch.sum(torch.pow(input_x_test[:,window_size-step_size:window_size,:]-test_rebuild, 2), dim=2), 0.5)
            score2 = torch.pow(
                torch.sum(torch.pow(input_x_test[:, window_size - step_size:window_size, :]*test_root - test_rebuild*test_root, 2), dim=2), 0.5)
            threshold1.append(score1)
            threshold2.append(score2)
            test_y.append(input_y_test[:,window_size-step_size:window_size,:])
    threshold1=torch.cat(threshold1)
    threshold2=torch.cat(threshold2)
    test_y=torch.cat(test_y).squeeze(2)


    thresholds_setting = np.percentile(threshold1.cpu().detach().numpy(), 100 * (
                1 - np.array([0.3, 0.2, 0.1, 0.08,0.05,0.02, 0.015, 0.01, 0.005, 0.002, 0.001, 0.0005, 0.0002, 0.0001])))
    precision, recall, acc, fpr = evalution_roc(thresholds_setting, threshold1, test_y)
    print(precision,recall)
    precision, recall=precision[np.where(acc == max(acc))[0]], recall[np.where(acc == max(acc))[0]]
    print(np.where(acc == max(acc))[0])
    print("Precision:",precision)
    print("Recall:",recall)
    print("F1:", 2 * recall * precision / (recall + precision))





