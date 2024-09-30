# -*- coding: utf-8 -*-
from datetime import datetime
import time 
import argparse
import copy
import torch
from torch import optim
from sklearn import metrics
import pandas as pd
import numpy as np
from sklearn.metrics import roc_auc_score, average_precision_score, f1_score,accuracy_score,recall_score,precision_score,roc_curve,auc,confusion_matrix
import newmodels
from data_preprocessing import DrugDataset, DrugDataset1,DrugDataLoader, TOTAL_ATOM_FEATS
import os
from torch.autograd import Variable
import sys
os.environ["CUDA_VISIBLE_DEVICES"] = "1"

######################### Parameters ######################
parser = argparse.ArgumentParser()
parser.add_argument('--n_atom_feats', type=int, default=TOTAL_ATOM_FEATS, help='num of input features')
parser.add_argument('--n_atom_hid', type=int, default=64, help='num of hidden features')
parser.add_argument('--rel_total', type=int, default=65, help='num of interaction types')
parser.add_argument('--lr', type=float, default=5e-3, help='learning rate')
parser.add_argument('--n_epochs', type=int, default=150, help='num of epochs')
parser.add_argument('--kge_dim', type=int, default=128, help='dimension of interaction matrix')
parser.add_argument('--batch_size', type=int, default=64, help='batch size')

parser.add_argument('--weight_decay', type=float, default=5e-4)
parser.add_argument('--neg_samples', type=int, default=1)
parser.add_argument('--data_size_ratio', type=int, default=1)
parser.add_argument('--use_cuda', type=bool, default=True, choices=[0,1])
parser.add_argument('--zhongzi', type=int, default=0)
parser.add_argument('--subnum', type=int, default=20)
parser.add_argument('--atomnum', type=int, default=60)


args = parser.parse_args()
atomnum=args.atomnum
n_atom_feats = args.n_atom_feats
n_atom_hid = args.n_atom_hid
rel_total = args.rel_total
lr = args.lr
n_epochs = args.n_epochs
kge_dim = args.kge_dim
batch_size = args.batch_size
zhongzi=args.zhongzi
weight_decay = args.weight_decay
neg_samples = args.neg_samples
data_size_ratio = args.data_size_ratio
sn = args.subnum
an = args.atomnum
#device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')
device = 'cuda' if torch.cuda.is_available() and args.use_cuda else 'cpu'
# lr=float(sys.argv[1])
# n_epochs=int(sys.argv[2])
# zhongzi=int(sys.argv[3])
print(args)

###### Dataset
df_ddi_train = pd.read_csv('data/'+str(zhongzi)+'/train.csv')
df_ddi_val = pd.read_csv('data/'+str(zhongzi)+'/valid.csv')
df_ddi_test = pd.read_csv('data/'+str(zhongzi)+'/test.csv')


train_tup = [(h, t, r) for h, t, r in zip(df_ddi_train['d1'], df_ddi_train['d2'], df_ddi_train['type'])]
val_tup = [(h, t, r) for h, t, r in zip(df_ddi_val['d1'], df_ddi_val['d2'], df_ddi_val['type'])]
test_tup = [(h, t, r) for h, t, r in zip(df_ddi_test['d1'], df_ddi_test['d2'], df_ddi_test['type'])]

train_data = DrugDataset(train_tup, ratio=data_size_ratio, neg_ent=neg_samples)
val_data = DrugDataset(val_tup, ratio=data_size_ratio, disjoint_split=False)
test_data = DrugDataset(test_tup, disjoint_split=False)
druglist=DrugDataset1(train_tup, ratio=data_size_ratio, neg_ent=neg_samples)

print(f"Training with {len(train_data)} samples, validating with {len(val_data)}, and testing with {len(test_data)}")

train_data_loader = DrugDataLoader(train_data, batch_size=batch_size, shuffle=True,drop_last=True)
val_data_loader = DrugDataLoader(val_data, batch_size=batch_size ,drop_last=True)
test_data_loader = DrugDataLoader(test_data, batch_size=1,drop_last=True)
druglist=DrugDataLoader(druglist)

# def compute_weigth(batch, device, training=True):
#     '''
#         *batch: (pos_tri, neg_tri)
#         *pos/neg_tri: (batch_h, batch_t, batch_r)
#     '''
#     probas_pred, ground_truth = [], []
#     for batch in druglist:
#         pos_tri = batch
#         pos_tri = [tensor.to(device=device) for tensor in pos_tri]
#
#         #h_weight = model.get_weight(pos_tri,zhongzi)
#         break
#
#     return h_weight

def do_compute(batch, device, model,type,training=True):
        '''
            *batch: (pos_tri, neg_tri)
            *pos/neg_tri: (batch_h, batch_t, batch_r)
        '''
        probas_pred, ground_truth = [], []
        pos_tri, neg_tri ,hlist= batch
        #print(pos_tri[0])
        # gt = []
        # #print(np.array(pos_tri[2]).shape)
        # for tri in pos_tri[2]:
        #
        #  #   print(tri.item())
        #     gt.append(tri.item())
        # gt = np.array(gt)

        pos_tri = [tensor.to(device=device) for tensor in pos_tri]
        #print(pos_tri[0][0])
        p_score,gt,i,d_v,p_v,a11,a22 = model(pos_tri)
        if type=='test':
            with open('testmap1.txt', 'a') as f:
                np.set_printoptions(threshold=sys.maxsize)
                f.write(str(i.detach().cpu().numpy()))
                f.write(str(d_v.detach().cpu().numpy()))
                f.write(str(p_v.detach().cpu().numpy()))
                f.write(str(a11.detach().cpu().numpy()))
                f.write(str(a22.detach().cpu().numpy()))
        gt = np.array(gt.cpu()).reshape(-1)
        gt = torch.tensor(gt).to(device).float()

        #p_score=p_score.detach().cpu().numpy()
        #probas_pred.append(torch.sigmoid(p_score.detach()).cpu())
        #probas_pred.append(p_score.detach().cpu().numpy())
        #ground_truth.append(gt)
        #ground_truth.append(np.ones(len(p_score)))

        # neg_tri = [tensor.to(device=device) for tensor in neg_tri]
        # n_score = model(neg_tri)
        # probas_pred.append(torch.sigmoid(n_score.detach()).cpu())
        # ground_truth.append(np.zeros(len(n_score)))
        #
        # probas_pred = np.concatenate(probas_pred)
        # ground_truth = np.concatenate(ground_truth)

        return torch.squeeze(p_score, 1),gt


def do_compute_metrics(probas_pred, target):
    # import copy
    # pred = copy.deepcopy(probas_pred)
    # for i in range(probas_pred.shape[0]):
    #     a=max(probas_pred[i])
    #     for k in range(probas_pred.shape[1]):
    #         if probas_pred[i][k]==a:
    #             probas_pred[i][k]=1
    #         else:
    #             probas_pred[i][k] = 0
    # pred = (probas_pred >= 0.5).astype(np.int)
    # acc = metrics.accuracy_score(target, pred)
    # auc_roc = metrics.roc_auc_score(target, probas_pred)
    # f1_score = metrics.f1_score(target, pred)

    #p, r, t = metrics.precision_recall_curve(target, probas_pred)
    #auc_prc = metrics.auc(r, p)

    y_pred_train1 =probas_pred
    y_label_train = np.array(target)
    # #print(np.array(probas_pred)[0:20])
    # #print(y_label_train.shape)
    # #print(probas_pred.shape)
    # y_label_train=y_label_train.reshape((-1))
    # y_pred_train = np.array(probas_pred).reshape((-1, 2))
    # # print(y_pred_train.shape)
    # for i in range(y_pred_train.shape[0]):
    #     a = np.max(y_pred_train[i])
    #     # print(y_pred_train[i])
    #     # print(a)
    #     for j in range(y_pred_train.shape[1]):
    #         if y_pred_train[i][j] == a:
    #             # print(y_pred_train[i][j])
    #             y_pred_train1.append(j)
    #             break

    #print(np.array(y_label_train[0:20]))
    #print(np.array(y_pred_train1[0:20]))
    #print(np.array(y_label_train).shape)
    #print(np.array(y_pred_train1).shape)

    # print(111111,y_pred_train1)
    # print(222222,y_label_train)
    fpr, tpr, thresholds = roc_curve(y_label_train, y_pred_train1)
    precision = tpr / (tpr + fpr)
    f1 = 2 * precision * tpr / (tpr + precision + 0.00001)
    thred_optim = thresholds[5:][np.argmax(f1[5:])]

    y_pred_s = [1 if i else 0 for i in (y_pred_train1 >= thred_optim)]
    cm1 = confusion_matrix(y_label_train, y_pred_s)
    total1 = sum(sum(cm1))
    #####from confusion matrix calculate accuracy
    acc = (cm1[0, 0] + cm1[1, 1]) / total1



    auc_k = auc(fpr, tpr)
    aupr=average_precision_score(y_label_train, y_pred_train1)
    outputs = np.asarray([1 if i else 0 for i in (np.asarray(y_pred_train1) >= 0.5)])
    #acc = accuracy_score(y_label_train, y_pred_train1)
    f1_score1 = f1_score(y_label_train, outputs)
    recall1 = recall_score(y_label_train, y_pred_s)
    precision1 = precision_score(y_label_train, y_pred_s)

    return auc_k,aupr,acc, f1_score1, recall1,precision1


def train(model, train_data_loader, val_data_loader, loss_fn,  optimizer, n_epochs, device, scheduler=None):
    print('Starting training at', datetime.today())
    m = torch.nn.Sigmoid()
    maxauc=0
    #maxaupr=0
    for i in range(1, n_epochs+1):
        start = time.time()
        train_loss = 0
        train_loss_pos = 0
        train_loss_neg = 0
        val_loss = 0
        val_loss_pos = 0
        val_loss_neg = 0
        train_probas_pred = []
        train_ground_truth = []
        val_probas_pred = []
        val_ground_truth = []

        for batch in train_data_loader:
            #with torch.no_grad():
            #    drug_weight = compute_weigth(druglist, device)
            model.train()
            #print(batch)
            p_score, gt= do_compute(batch, device,model,'train')
            # print(p_score)
            # print(gt)
            #p_score=torch.FloatTensor(p_score,requires_grad=False).to_device('cuda')
            #label = np.array(gt, dtype=np.int64)
            #label = torch.from_numpy(label)

            #gt = label.cuda()
            #gt=torch.LongTensor(gt,requires_grad=False)#.to_device('cuda')
            #p_score=p_score
            #print(p_score[0])
            #print(gt[0])

            loss = loss_fn(p_score, gt)
            # train_ground_truth1 = torch.tensor(train_ground_truth, dtype=torch.long, requires_grad=False)
            # train_probas_pred1 = torch.tensor(train_probas_pred, requires_grad=True)
            #print(p_score)
            m = torch.nn.Sigmoid()
            p_score= torch.squeeze(m(p_score))
            p_score=p_score.detach().cpu().numpy()
            gt=gt.detach().cpu().numpy()

            train_probas_pred.append(np.array(p_score))
            train_ground_truth.append(gt)
            #
            # train_ground_truth1=torch.tensor(train_ground_truth,dtype=torch.long,requires_grad=False)
            # train_probas_pred1=torch.tensor(train_probas_pred,requires_grad=True)
            #
            # train_ground_truth1=train_ground_truth1.view(-1)
            # train_probas_pred1=train_probas_pred1.view((-1,86))
            # #print(train_probas_pred.shape)
            # #print(train_ground_truth.shape)
            # loss=loss_fn(train_probas_pred1,train_ground_truth1.long())
            #loss, loss_p, loss_n = loss_fn(p_score, n_score)
            
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            # s =model.rel_emb.grad
            # print(s)
            # # s = model.set2set.grad
            # # print(s)
            # s = model.lin0.grad
            # print(s)
            # s = model.convs.grad
            # print(s)

            train_loss += loss.item() * len(p_score)


        train_loss /= len(train_data)

        with torch.no_grad():
            train_probas_pred = np.concatenate(train_probas_pred)
            train_ground_truth = np.concatenate(train_ground_truth)
            #print(train_probas_pred.shape)
            #print(train_ground_truth.shape)
            train_auc,train_aupr,train_acc, train_f1, train_recall,train_precision = do_compute_metrics(train_probas_pred, train_ground_truth)
            # drug_weight=compute_weigth(druglist,device)

            for batch in val_data_loader:
                model.eval()
                probas_pred, gt,  = do_compute(batch, device,model,'valid',training=False)
                # val_ground_truth1 = val_ground_truth1.view(-1)
                # val_probas_pred1 = val_probas_pred1.view((-1, 86))
                #print(probas_pred.shape)
                #print(gt.shape)
                #label = np.array(gt, dtype=np.int64)
                #label = torch.from_numpy(label)

                #gt = label.cuda()
                #probas_pred = m(probas_pred)
                loss = loss_fn(probas_pred, gt)
                m = torch.nn.Sigmoid()
                probas_pred = torch.squeeze(m(probas_pred))
                probas_pred = probas_pred.detach().cpu().numpy()
                gt = gt.detach().cpu().numpy()

                val_probas_pred.append(np.array(probas_pred))
                val_ground_truth.append(gt)
                #
                # val_ground_truth1 = torch.tensor(val_ground_truth, dtype=torch.long, requires_grad=False)
                # val_probas_pred1 = torch.tensor(val_probas_pred, requires_grad=True)
                # #print(val_probas_pred1.shape)
                # #print(val_ground_truth1.shape)
                # val_ground_truth1 = val_ground_truth1.view(-1)
                # val_probas_pred1 = val_probas_pred1.view((-1, 86))
                # loss = loss_fn(val_probas_pred1, val_ground_truth1.long())
                val_loss += loss.item() * len(probas_pred)

            val_loss /= len(val_data)
            val_probas_pred = np.concatenate(val_probas_pred)
            val_ground_truth = np.concatenate(val_ground_truth)
            val_auc,val_aupr,val_acc, val_f1, val_recall, val_precision = do_compute_metrics(val_probas_pred, val_ground_truth)
            if scheduler:
            # print('scheduling')
                scheduler.step()
            if val_auc >= maxauc:
                #and val_aupr >= maxaupr:
                model_max = copy.deepcopy(model)
                maxauc = val_auc
                #maxaupr=val_aupr
            #else:  #
             #   model_max = copy.deepcopy(model)

            # print('epoch: {:04d}'.format(epoch + 1),
            #       'loss_train: {:.4f}'.format(loss_train.item()),
            #       'auroc_train: {:.4f}'.format(acc),
            #       'loss_val: {:.4f}'.format(loss_val.item()),
            #       'acc_val: {:.4f}'.format(acc_val),
            #       'f1_val: {:.4f}'.format(f1_val),
            #       'recall_val: {:.4f}'.format(recall_val),
            #       'precision_val: {:.4f}'.format(precision_val),
            #       'time: {:.4f}s'.format(time.time() - t))




        print(f'Epoch: {i} ({time.time() - start:.4f}s), train_loss: {train_loss:.4f}, val_loss: {val_loss:.4f},'f' val_auc: {val_auc:.4f}, val_aupr:{val_aupr:.4f}')
        #print(f'\t\ttrain_f1: {train_f1:.4f}, val_f1: {val_f1:.4f}, train_val_recall: {train_recall:.4f}, val_recall: {val_recall:.4f}, train_precision: {train_precision:.4f}, val_precision: {val_precision:.4f}')

    test_probas_pred=[]
    test_ground_truth=[]
    test_loss=0
    with torch.no_grad():
        for batch in test_data_loader:
            model.eval()
            print("111111111")
            probas_pred, gt, = do_compute(batch, device,model_max,'test')
            print("222222222")
            loss = loss_fn(probas_pred, gt)
            m = torch.nn.Sigmoid()
            probas_pred = torch.squeeze(m(probas_pred))
            probas_pred = probas_pred.detach().cpu().numpy()
            gt = gt.detach().cpu().numpy()

            test_probas_pred.append(np.array(probas_pred))
            test_ground_truth.append(gt)

            test_loss += loss.item() * len(probas_pred)
        #
        # test_loss /= len(test_data)
        # test_probas_pred = np.concatenate(test_probas_pred)
        # test_ground_truth = np.concatenate(test_ground_truth)
        # test_auc,test_aupr,test_acc, test_f1, test_recall, test_precision = do_compute_metrics(test_probas_pred, test_ground_truth)

    if scheduler:
        # print('scheduling')
        scheduler.step()

    # print(f'Epoch: {i} ({time.time() - start:.4f}s), train_loss: {train_loss:.4f}, val_loss: {val_loss:.4f},'
    #       f' train_acc: {train_acc:.4f}, val_acc:{val_acc:.4f}')
    # print(f'\t\t test_acc: {test_acc:.4f}, test_f1: {test_f1:.4f}, test_recall: {test_recall:.4f},test_precision: {test_precision:.4f}')
    # with open('mulnet11.txt', 'a') as f:
    #     # f.write('{0}\t{1}\t{2}\t{7}\t{3:.4f}\t{4:.4f}\t{5:.4f}\t{6:.4f}\n'.format(
    #     #     args.in_file[5:8], args.seed, args.aggregator, loss_test.item(), acc_test, f1_test, recall_test, args.feature_type))
    #     f.write(str(atomnum)+' '+str(n_epochs)+' '+str(zhongzi)+'  '+str(test_auc)+'  '+str(test_aupr)+' '+str(test_acc)+'  '+str(test_f1)+'  '+str(test_recall)+'  '+str(test_precision)+'\n')

#model = models.SSI_DDI(n_atom_feats, n_atom_hid, kge_dim, rel_total, heads_out_feat_params=[32, 32, 32, 32], blocks_params=[2, 2, 2, 2])
#
#model = models.TrimNet(55, 10, hidden_dim=64, depth=3,heads=4, dropout=0.2, outdim=1)
#model=models.NNCNN(55,55)
model=newmodels.BaseGGNN(128, 3,subnum=sn,atomnum=an)
#n=[n for n, p in model.named_parameters()]
#print(n)
#model.load_state_dict(torch.load(args.model_path, map_location='cuda:0'))
model.to(device=device)
#loss = custom_loss.SigmoidLoss()
loss=torch.nn.BCEWithLogitsLoss()
#loss=torch.nn.NLLLoss()
optimizer = optim.Adam(model.parameters(), lr=lr, weight_decay=weight_decay)
scheduler = optim.lr_scheduler.LambdaLR(optimizer, lambda epoch: 0.96 ** (epoch))
# print(model)


# if __name__ == '__main__':
train(model, train_data_loader, val_data_loader, loss, optimizer, n_epochs, device, scheduler)


