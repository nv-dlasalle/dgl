import pandas as pd
import torch as th
import numpy as np
from tqdm import tqdm
import sklearn.metrics as skm
import dgl
import torch.nn.functional as F
from torch import nn
import argparse

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

def load_oag(dataset):
    dataset = dgl.load_graphs(dataset)[0]
    hg = dataset[0]

    node_feats = []
    for ntype in hg.ntypes:
        if 'emb' in hg.nodes[ntype].data:
            feat = hg.nodes[ntype].data.pop('emb')
            node_feats.append(feat.share_memory_())
        else:
            node_feats.append(None)

    # Construct labels of paper nodes
    ss, dd = hg.edges(etype=('field', 'rev_PF_in_L1', 'paper'))
    ssu_, ssu = th.unique(ss, return_inverse=True)
    print('Full label set size:', len(ssu_))
    paper_labels = th.zeros(hg.num_nodes('paper'), len(ssu_), dtype=th.bool)
    paper_labels[dd, ssu] = True

    # Split the dataset into training, validation and testing.
    label_sum = paper_labels.sum(1)
    valid_labal_idx = th.nonzero(label_sum > 0, as_tuple=True)[0]
    train_size = int(len(valid_labal_idx) * 0.8)
    val_size = int(len(valid_labal_idx) * 0.1)
    test_size = len(valid_labal_idx) - train_size - val_size
    train_idx, val_idx, test_idx = valid_labal_idx[th.randperm(len(valid_labal_idx))].split([train_size, val_size, test_size])

    # Remove infrequent labels. Otherwise, some of the labels will not have instances
    # in the training, validation or test set.
    label_filter = paper_labels[train_idx].sum(0) > 100
    label_filter = th.logical_and(label_filter, paper_labels[val_idx].sum(0) > 100)
    label_filter = th.logical_and(label_filter, paper_labels[test_idx].sum(0) > 100)
    paper_labels = paper_labels[:,label_filter]
    print('#labels:', paper_labels.shape[1])

    # Adjust training, validation and testing set to make sure all paper nodes
    # in these sets have labels.
    train_idx = train_idx[paper_labels[train_idx].sum(1) > 0]
    val_idx = val_idx[paper_labels[val_idx].sum(1) > 0]
    test_idx = test_idx[paper_labels[test_idx].sum(1) > 0]
    # All labels have instances.
    assert np.all(paper_labels[train_idx].sum(0).numpy() > 0)
    assert np.all(paper_labels[val_idx].sum(0).numpy() > 0)
    assert np.all(paper_labels[test_idx].sum(0).numpy() > 0)
    # All instances have labels.
    assert np.all(paper_labels[train_idx].sum(1).numpy() > 0)
    assert np.all(paper_labels[val_idx].sum(1).numpy() > 0)
    assert np.all(paper_labels[test_idx].sum(1).numpy() > 0)
    category = 'paper'
    return hg, node_feats, paper_labels, train_idx, val_idx, test_idx, category, paper_labels.shape[1]

def config():
    parser = argparse.ArgumentParser(description='MLP')
    parser.add_argument("--n-hidden", type=int, default=512,
            help="number of hidden units")
    parser.add_argument("--gpu", type=str, default='0',
            help="gpu")
    parser.add_argument("--lr", type=float, default=1e-2,
            help="learning rate")
    parser.add_argument("-e", "--n-epochs", type=int, default=50,
            help="number of training epochs")
    parser.add_argument("-d", "--dataset", type=str, required=True,
            help="dataset to use")
    parser.add_argument("--batch-size", type=int, default=100,
            help="Mini-batch size. ")
    parser.add_argument("--num-workers", type=int, default=0,
            help="Number of workers for dataloader.")
    parser.set_defaults(validation=True)
    args = parser.parse_args()
    return args


def train(net, dataloader, val_dl, features, labels, lr, num_epochs):
    opt = th.optim.Adam(net.parameters(), lr=lr)

    ps = labels.sum(0).float()
    pw = (labels.shape[0] - ps) / ps

    for epoch in range(num_epochs):
        losses = []
        with tqdm(dataloader) as tq:
            for batch in tq:
                opt.zero_grad()
                x = features[batch]
                y_hat = net(x)
                y = labels[batch].float()
                loss = F.binary_cross_entropy_with_logits(y_hat, y, pos_weight=pw)
                loss.backward()
                opt.step()
                losses.append(loss.item())
        print('epoch {} | loss {}'.format(epoch, np.mean(losses)))
        if epoch % 5 == 0 and epoch > 0:
            evaluate(net, val_dl, features, labels)
    if num_epochs % 5 != 0:
        evaluate(net, val_dl, features, labels)


def baseline(dataloader, paper_labels):
    ys = []
    with tqdm(dataloader) as tq:
        for batch in tq:
            ys.append(paper_labels[batch].float().numpy())
    ys = np.concatenate(ys)
    print('Random baseline:',
          skm.roc_auc_score(ys, np.random.randn(*ys.shape)),
          skm.average_precision_score(ys, np.random.randn(*ys.shape)))

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
        ys = th.cat(ys, 0).cpu().numpy()
        y_hats = th.cat(y_hats, 0).cpu().numpy()
        print('AUC:', skm.roc_auc_score(ys, y_hats))
        print('APS', skm.average_precision_score(ys, y_hats))


def main(args):
    graph, node_feats, paper_labels, train_idx, val_idx, test_idx, category, \
        num_classes = load_oag(args.dataset)
    paper_feats = node_feats[graph.get_ntype_id('paper')]

    train_dl = th.utils.data.DataLoader(train_idx, batch_size=args.batch_size, shuffle=True, drop_last=False)
    val_dl = th.utils.data.DataLoader(val_idx, batch_size=args.batch_size, shuffle=True, drop_last=False)
    test_dl = th.utils.data.DataLoader(test_idx, batch_size=args.batch_size, shuffle=True, drop_last=False)

    print("Validation set")
    baseline(val_dl, paper_labels)

    print("Test set")
    baseline(test_dl, paper_labels)

    net = MLP(paper_feats.shape[1], args.n_hidden, num_classes)
    net = net.cuda()

    paper_labels = paper_labels.cuda()
    paper_feats = paper_feats.cuda()

    train(net, train_dl, val_dl, paper_feats, paper_labels, args.lr, args.n_epochs)
    evaluate(net, test_dl, paper_feats, paper_labels)

if __name__ == '__main__':
    args = config()
    main(args)
