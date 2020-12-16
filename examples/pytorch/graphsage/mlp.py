import pandas as pd
import torch as th
import numpy as np
from tqdm import tqdm
import sklearn.metrics as skm
from sklearn.preprocessing import normalize
import dgl
import torch.nn.functional as F
from torch import nn
import argparse
from load_graph import load_ogb, load_reddit, inductive_split

class MLP(nn.Module):
    def __init__(self, in_feats, n_hiddens, out_feats):
        super().__init__()
        self.W1 = nn.Linear(in_feats, n_hiddens)
        self.W2 = nn.Linear(n_hiddens, n_hiddens)
        self.W3 = nn.Linear(n_hiddens, out_feats)
        
    def forward(self, x):
        x = F.relu(self.W1(x))
        x = F.relu(self.W2(x))
        return self.W3(x)

def config():
    parser = argparse.ArgumentParser(description='MLP')
    parser.add_argument("--n-hidden", type=int, default=128,
            help="number of hidden units")
    parser.add_argument("--gpu", type=str, default='0',
            help="gpu")
    parser.add_argument("--lr", type=float, default=1e-3,
            help="learning rate")
    parser.add_argument("-e", "--n-epochs", type=int, default=50,
            help="number of training epochs")
    parser.add_argument("-d", "--dataset", type=str, required=True,
            help="dataset to use")
    parser.add_argument("--batch-size", type=int, default=1000,
            help="Mini-batch size. ")
    parser.add_argument("--num-workers", type=int, default=0,
            help="Number of workers for dataloader.")
    parser.set_defaults(validation=True)
    args = parser.parse_args()
    return args


def train(net, dataloader, val_dl, features, labels, lr, num_epochs):
    opt = th.optim.Adam(net.parameters(), lr=lr)

    loss_fcn = nn.CrossEntropyLoss()
    loss_fcn = loss_fcn.cuda()

    for epoch in range(num_epochs):
        losses = []
        with tqdm(dataloader) as tq:
            for batch in tq:
                opt.zero_grad()
                x = features[batch]
                y_hat = net(x)
                y = labels[batch]
                loss = loss_fcn(y_hat, y)
                loss.backward()
                opt.step()
                losses.append(loss.item())
        print('epoch {} | loss {}'.format(epoch, np.mean(losses)))
        if epoch % 5 == 0 and epoch > 0:
            evaluate(net, val_dl, features, labels)
    if num_epochs % 5 != 0:
        evaluate(net, val_dl, features, labels)


def compute_acc(labels, pred):
#    skm.roc_auc_score(labels.cpu().numpy(), pred.cpu().numpy(),
#          multi_class='ovo', labels=range(pred.shape[1])),
#    skm.average_precision_score(labels.cpu().numpy(), pred.cpu().numpy()))
    return (th.argmax(pred, dim=1) == labels).float().sum() / labels.shape[0] 

def baseline(dataloader, paper_labels, n_classes):
    ys = []
    with tqdm(dataloader) as tq:
        for batch in tq:
            ys.append(paper_labels[batch])
    ys = th.cat(ys)
    y_hat = th.abs(th.randn((ys.shape[0], n_classes), device=th.device(0)))
    y_hat = y_hat / y_hat.sum(dim=1, keepdim=True)
    print('Random baseline:', compute_acc(ys, y_hat))

def evaluate(net, dataloader, features, labels):
    with tqdm(dataloader) as tq, th.no_grad():
        y_hats = []
        ys = []
        for batch in tq:
            x = features[batch]
            y = labels[batch]
            y_hat = net(x)
            y_hats.append(y_hat)
            ys.append(y)
        ys = th.cat(ys, 0)
        y_hats = th.cat(y_hats, 0)
        print("ACC:", compute_acc(ys, y_hats))


def main(args):
    g, n_classes = load_ogb(args.dataset)
    node_feats = g.ndata['features']
    labels = g.ndata['labels']

    train_g, val_g, test_g = inductive_split(g)

    train_mask = train_g.ndata['train_mask']
    val_mask = val_g.ndata['val_mask']
    test_mask = ~(test_g.ndata['train_mask'] | test_g.ndata['val_mask'])
    train_idx = train_mask.nonzero().squeeze()
    val_idx = val_mask.nonzero().squeeze()
    test_idx = test_mask.nonzero().squeeze()

    train_dl = th.utils.data.DataLoader(train_idx, batch_size=args.batch_size, shuffle=True, drop_last=False)
    val_dl = th.utils.data.DataLoader(val_idx, batch_size=args.batch_size, shuffle=True, drop_last=False)
    test_dl = th.utils.data.DataLoader(test_idx, batch_size=args.batch_size, shuffle=True, drop_last=False)

    labels = labels.cuda()
    node_feats = node_feats.cuda()

    print("Validation set")
    baseline(val_dl, labels, n_classes)
    print("Test set")
    baseline(test_dl, labels, n_classes)

    net = MLP(node_feats.shape[1], args.n_hidden, n_classes)
    net = net.cuda()

    train(net, train_dl, val_dl, node_feats, labels, args.lr, args.n_epochs)
    evaluate(net, test_dl, node_feats, labels)

if __name__ == '__main__':
    args = config()
    main(args)
