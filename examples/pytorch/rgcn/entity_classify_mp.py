"""
Modeling Relational Data with Graph Convolutional Networks
Paper: https://arxiv.org/abs/1703.06103
Code: https://github.com/tkipf/relational-gcn
Difference compared to tkipf/relation-gcn
* l2norm applied to all weights
* remove nodes that won't be touched
"""
import argparse
import itertools
import numpy as np
import time
import gc
import torch as th
import torch.nn as nn
import torch.nn.functional as F
import torch.multiprocessing as mp
from torch.cuda import nvtx
from torch.multiprocessing import Queue
from torch.nn.parallel import DistributedDataParallel
from torch.utils.data import DataLoader
import dgl
from dgl import DGLGraph
from dgl.dataloading import AsyncTransferer
from functools import partial
import os
import subprocess

from dgl.data.rdf import AIFBDataset, MUTAGDataset, BGSDataset, AMDataset
from model import RelGraphEmbedLayer, RelFeatLayer
from dgl.nn import RelGraphConv
from utils import thread_wrapped_func
import tqdm 

from ogb.nodeproppred import DglNodePropPredDataset

class PrefetchingIterator:
    def __init__(self, i, dev_id, g, feats, labels, num_of_ntype):
        self.iter_ = i
        self.dev_id_ = dev_id
        self.g_ = g
        self.feats_ = feats
        self.labels_ = labels
        self.num_of_ntype_ = num_of_ntype
        self.async_ = AsyncTransferer(dev_id)
        self.fetched_ = None
        self.multilabel_ = multilabel = len(self.labels_.shape) > 1

    def get_next(self):
        ret = None
        try:
            nvtx.range_push("NodeDataLoader.next()")
            ret = next(self.iter_)
        finally:
            nvtx.range_pop()
        return ret
            

    def __next__(self):
        if self.fetched_ is None:
            try:
                nvtx.range_push("__next__.no_prefetch")
                # directly load everything
                seeds, blocks = self.get_next()

                labels = self.labels_[seeds].to(self.dev_id_, non_blocking=True)
                labels = labels.float() if self.multilabel_ else labels.long()
                loc = [None for i in range(self.num_of_ntype_)]
                features = [None for i in range(self.num_of_ntype_)]

                node_ids_cpu = blocks[0].srcdata[dgl.NID] 
                blocks_gpu = [block.to(self.dev_id_, non_blocking=True) for block in blocks]
                # pull in block data too
                for block in blocks_gpu:
                    block.edata['etype']
                    block.edata['norm']

                for ntype in range(self.num_of_ntype_):
                    loc[ntype] = (blocks[0].srcdata[dgl.NTYPE] == ntype).nonzero().pin_memory().squeeze(-1)
                    if self.feats_[ntype] is not None:
                        feats_cpu = th.empty((loc[ntype].shape[0], self.feats_[ntype].shape[1]), pin_memory=True)
                        th.index_select(self.feats_[ntype], 0,
                            blocks[0].srcdata['type_id'][loc[ntype]],
                            out=feats_cpu)
                        features[ntype] = feats_cpu.to(self.dev_id_, non_blocking=True)
            finally:
                nvtx.range_pop()

        if self.dev_id_ != 'cpu':
            nvtx.range_push("prefetch")
            # initiate next fetch
            next_fetch = None
            try:
                next_seeds, next_blocks = self.get_next()

                next_labels = self.labels_[next_seeds].to(self.dev_id_, non_blocking=True)
                next_labels = next_labels.float() if self.multilabel_ else next_labels.long()
                next_loc = [None for i in range(self.num_of_ntype_)]
                next_features = [None for i in range(self.num_of_ntype_)]
                next_node_ids_cpu = next_blocks[0].srcdata[dgl.NID] 

                next_blocks_gpu = [block.to(self.dev_id_, non_blocking=True) for block in next_blocks]
                # pull in block data too
                for block in next_blocks_gpu:
                    block.edata['etype']
                    block.edata['norm']

                for ntype in range(self.num_of_ntype_):
                    next_loc[ntype] = (next_blocks[0].srcdata[dgl.NTYPE] == ntype).nonzero().pin_memory().squeeze(-1)

                desc_ntype = [*range(self.num_of_ntype_)]
                desc_ntype.sort(reverse=True, key=lambda i:
                    self.feats_[i].shape[1]*next_loc[i].shape[0] if self.feats_[i] is not None else 0
                )
                for ntype in desc_ntype:
                    if self.feats_[ntype] is not None:
                        next_feats_cpu = th.empty((next_loc[ntype].shape[0], self.feats_[ntype].shape[1]), pin_memory=True)
                        th.index_select(self.feats_[ntype], 0,
                            next_blocks[0].srcdata['type_id'][next_loc[ntype]],
                            out=next_feats_cpu)

                        next_features[ntype] = self.async_.async_copy(next_feats_cpu, self.dev_id_)
                
                next_fetch = (next_blocks_gpu, next_features, next_loc,
                    next_node_ids_cpu, next_labels)
            except StopIteration:
                # nothing to fetch
                pass

            if self.fetched_:
                # grap futures
                blocks_gpu, features_future, loc, node_ids_cpu, labels = self.fetched_
                self.fetched_ = None
                features = [None for f in features_future]
                for i in range(len(features)):
                    f = features_future[i]
                    if f is not None:
                        features[i] = f.wait()
            self.fetched_ = next_fetch
            nvtx.range_pop()

        return (blocks_gpu, features, loc, node_ids_cpu, labels)
        

class PrefetchingNodeDataLoader:
    def __init__(self, dev_id, g, feats, labels, num_of_ntype, dataset, sampler, **kwargs):
        self.dev_id_ = dev_id
        self.g_ = g
        self.feats_ = feats
        self.labels_ = labels
        self.num_of_ntype_ = num_of_ntype
        self.dataloader_ = DataLoader(
            dataset=dataset,
            collate_fn=sampler.sample_blocks,
            **kwargs)

    def __iter__(self):
        """Return the iterator of the data loader."""
        nvtx.range_push("create_node_iterator")
        it = self.dataloader_.__iter__()
        nvtx.range_pop()
        nvtx.range_push("create_prefetching_iterator")
        ret = PrefetchingIterator(it, self.dev_id_, self.g_, self.feats_,
            self.labels_, self.num_of_ntype_)
        nvtx.range_pop()
        return ret

    def __len__(self):
        """Return the number of batches of the data loader."""
        return len(self.dataloader_)



class EntityClassify(nn.Module):
    """ Entity classification class for RGCN
    Parameters
    ----------
    device : int
        Device to run the layer.
    num_nodes : int
        Number of nodes.
    h_dim : int
        Hidden dim size.
    out_dim : int
        Output dim size.
    num_rels : int
        Numer of relation types.
    num_bases : int
        Number of bases. If is none, use number of relations.
    num_hidden_layers : int
        Number of hidden RelGraphConv Layer
    dropout : float
        Dropout
    use_self_loop : bool
        Use self loop if True, default False.
    low_mem : bool
        True to use low memory implementation of relation message passing function
        trade speed with memory consumption
    """
    def __init__(self,
                 device,
                 num_nodes,
                 h_dim,
                 out_dim,
                 num_rels,
                 num_bases=None,
                 num_hidden_layers=1,
                 dropout=0,
                 use_self_loop=False,
                 low_mem=False,
                 layer_norm=False):
        super(EntityClassify, self).__init__()
        self.device = th.device(device if device >= 0 else 'cpu')
        self.num_nodes = num_nodes
        self.h_dim = h_dim
        self.out_dim = out_dim
        self.num_rels = num_rels
        self.num_bases = None if num_bases < 0 else num_bases
        self.num_hidden_layers = num_hidden_layers
        self.dropout = dropout
        self.use_self_loop = use_self_loop
        self.low_mem = low_mem
        self.layer_norm = layer_norm

        self.layers = nn.ModuleList()
        # i2h
        self.layers.append(RelGraphConv(
            self.h_dim, self.h_dim, self.num_rels, "basis",
            self.num_bases, activation=F.relu, self_loop=self.use_self_loop,
            low_mem=self.low_mem, dropout=self.dropout, layer_norm = layer_norm))
        # h2h
        for idx in range(self.num_hidden_layers):
            self.layers.append(RelGraphConv(
                self.h_dim, self.h_dim, self.num_rels, "basis",
                self.num_bases, activation=F.relu, self_loop=self.use_self_loop,
                low_mem=self.low_mem, dropout=self.dropout, layer_norm = layer_norm))
        # h2o
        self.layers.append(RelGraphConv(
            self.h_dim, self.out_dim, self.num_rels, "basis",
            self.num_bases, activation=None,
            self_loop=self.use_self_loop,
            low_mem=self.low_mem, layer_norm = layer_norm))

    def forward(self, blocks, feats, norm=None):
        if blocks is None:
            # full graph training
            blocks = [self.g] * len(self.layers)
        h = feats
        i = 0
        for layer, block in zip(self.layers, blocks):
            nvtx.range_push("forward.layer_{}".format(i))
            h = layer(block, h, block.edata['etype'], block.edata['norm'])
            nvtx.range_pop()
            i+=1
        return h

class NeighborSampler:
    """Neighbor sampler
    Parameters
    ----------
    g : DGLHeterograph
        Full graph
    target_idx : tensor
        The target training node IDs in g
    fanouts : list of int
        Fanout of each hop starting from the seed nodes. If a fanout is None,
        sample full neighbors.
    """
    def __init__(self, g, target_idx, fanouts):
        self.g = g
        self.target_idx = target_idx
        self.fanouts = fanouts

    """Do neighbor sample
    Parameters
    ----------
    seeds :
        Seed nodes
    Returns
    -------
    tensor
        Seed nodes, also known as target nodes
    blocks
        Sampled subgraphs
    """
    def sample_blocks(self, seeds):
        blocks = []
        etypes = []
        norms = []
        ntypes = []
        seeds = th.tensor(seeds).long()
        cur = self.target_idx[seeds]
        for fanout in self.fanouts:
            if fanout is None or fanout == -1:
                frontier = dgl.in_subgraph(self.g, cur)
            else:
                frontier = dgl.sampling.sample_neighbors(self.g, cur, fanout)
            etypes = self.g.edata[dgl.ETYPE][frontier.edata[dgl.EID]]
            block = dgl.to_block(frontier, cur)
            block.srcdata[dgl.NTYPE] = self.g.ndata[dgl.NTYPE][block.srcdata[dgl.NID]]
            block.srcdata['type_id'] = self.g.ndata[dgl.NID][block.srcdata[dgl.NID]]
            block.edata['etype'] = etypes
            cur = block.srcdata[dgl.NID]
            blocks.insert(0, block)
        return seeds, blocks

def evaluate(model, embed_layer, feat_layer, eval_loader, node_feats):
    model.eval()
    embed_layer.eval()
    eval_logits = []
    eval_seeds = []
 
    with th.no_grad():
        for sample_data in tqdm.tqdm(eval_loader):
            th.cuda.empty_cache()
            seeds, blocks = sample_data
            feats = embed_layer(blocks[0].srcdata[dgl.NID],
                    blocks[0].srcdata[dgl.NTYPE],
                    blocks[0].srcdata['type_id'],
                    node_feats)
            logits = model(blocks, feats)
            eval_logits.append(logits.cpu().detach())
            eval_seeds.append(seeds.cpu().detach())
    eval_logits = th.cat(eval_logits)
    eval_seeds = th.cat(eval_seeds)
 
    return eval_logits, eval_seeds


@thread_wrapped_func
def run(proc_id, n_gpus, args, devices, dataset, split, queue=None):
    dev_id = devices[proc_id] if devices[proc_id] != 'cpu' else -1
    g, node_feats, num_of_ntype, num_classes, num_rels, target_idx, \
        train_idx, val_idx, test_idx, labels = dataset
    if split is not None:
        train_seed, val_seed, test_seed = split
        train_idx = train_idx[train_seed]
        val_idx = val_idx[val_seed]
        test_idx = test_idx[test_seed]

    fanouts = [int(fanout) for fanout in args.fanout.split(',')]
    node_tids = g.ndata[dgl.NTYPE]
    sampler = NeighborSampler(g, target_idx, fanouts)

    loader = PrefetchingNodeDataLoader(
        devices[proc_id],
        g,
        node_feats,
        labels,
        num_of_ntype,
        dataset=train_idx,
        sampler=sampler,
        batch_size=args.batch_size,
        shuffle=True,
        drop_last=False,
        num_workers=args.num_workers,
        persistent_workers=True)



    # validation sampler
#    val_sampler = NeighborSampler(g, target_idx, [None] * args.n_layers)
#    val_loader = DataLoader(dataset=val_idx.numpy(),
#                            batch_size=args.eval_batch_size,
#                            collate_fn=val_sampler.sample_blocks,
#                            shuffle=False,
#                            num_workers=args.num_workers)
#
#    # validation sampler
#    test_sampler = NeighborSampler(g, target_idx, [None] * args.n_layers)
#    test_loader = DataLoader(dataset=test_idx.numpy(),
#                             batch_size=args.eval_batch_size,
#                             collate_fn=test_sampler.sample_blocks,
#                             shuffle=False,
#                             num_workers=args.num_workers)

    if n_gpus > 1:
        dist_init_method = 'tcp://{master_ip}:{master_port}'.format(
            master_ip='127.0.0.1', master_port='12345')
        world_size = n_gpus
        backend = 'nccl'

        # using sparse embedding or usig mix_cpu_gpu model (embedding model can not be stored in GPU)
        if dev_id < 0:
            backend = 'gloo'
        th.distributed.init_process_group(backend=backend,
                                          init_method=dist_init_method,
                                          world_size=world_size,
                                          rank=proc_id)

    # node features
    # None for one-hot feature, if not none, it should be the feature tensor.
    # 
    embed_layer = RelGraphEmbedLayer(dev_id,
                                     g.number_of_nodes(),
                                     node_tids,
                                     num_of_ntype,
                                     node_feats,
                                     args.n_hidden,
                                     sparse_emb=args.sparse_embedding)

    feat_layer = RelFeatLayer(dev_id, g.number_of_nodes(), num_of_ntype, node_feats, args.n_hidden)

    # create model
    # all model params are in device.
    model = EntityClassify(dev_id,
                           g.number_of_nodes(),
                           args.n_hidden,
                           num_classes,
                           num_rels,
                           num_bases=args.n_bases,
                           num_hidden_layers=args.n_layers - 2,
                           dropout=args.dropout,
                           use_self_loop=args.use_self_loop,
                           low_mem=args.low_mem,
                           layer_norm=args.layer_norm)

    multilabel = len(labels.shape) > 1
    loss_func = nn.BCEWithLogitsLoss() if multilabel else nn.CrossEntropyLoss()
    loss_func.to(th.device(dev_id if dev_id >= 0 else 'cpu'))

    if dev_id >= 0 and n_gpus == 1:
        th.cuda.set_device(dev_id)
        labels = labels.to(dev_id)
        model.cuda(dev_id)
        # embedding layer may not fit into GPU, then use mix_cpu_gpu
        if args.mix_cpu_gpu is False:
            embed_layer.cuda(dev_id)

    if n_gpus > 1:
        if dev_id < 0:
            feat_layer = DistributedDataParallel(feat_layer, device_ids=None, output_device=None)
            model = DistributedDataParallel(model, device_ids=None, output_device=None)
        else:
            labels = labels.to(dev_id)
            model.cuda(dev_id)
            feat_layer.cuda(dev_id)
            feat_layer = DistributedDataParallel(feat_layer, device_ids=[dev_id], output_device=dev_id)
            model = DistributedDataParallel(model, device_ids=[dev_id], output_device=dev_id)

    # optimizer
    dense_params = list(model.parameters())
    if args.node_feats:
        if  n_gpus > 1:
            dense_params += list(feat_layer.module.embeds.parameters())
        else:
            dense_params += list(feat_layer.embeds.parameters())
    optimizer = th.optim.Adam(dense_params, lr=args.lr, weight_decay=args.l2norm)

    # training loop
    print("start training...")
    forward_time = []
    backward_time = []

    for epoch in range(args.n_epochs):
        model.train()
        feat_layer.train()

        loss = []
        tic = time.time()
        for i, sample_data in enumerate(loader):
            blocks_gpu, features_gpu, loc_cpu, node_ids_cpu, labels = sample_data
            t0 = time.time()

            tmp_feats = feat_layer(features_gpu)
            feats = th.empty(node_ids_cpu.shape[0], args.n_hidden, device=devices[proc_id])
            for ntype in range(num_of_ntype):
                if node_feats[ntype] is not None:
                    loc_gpu = loc_cpu[ntype].to(devices[proc_id], non_blocking=True)
                    feats[loc_gpu] = tmp_feats[ntype]

            nvtx.range_push("forward")
            logits = model(blocks_gpu, feats)
            nvtx.range_pop()

            nvtx.range_push("F.cross_entropy")
            loss = loss_func(logits, labels)
            nvtx.range_pop()
            t1 = time.time()

            nvtx.range_push("zero_grad")
            optimizer.zero_grad()
            nvtx.range_pop()

            nvtx.range_push("loss.backward")
            loss.backward()
            nvtx.range_pop()

            nvtx.range_push("optimizer.step")
            optimizer.step()
            nvtx.range_pop()

            t2 = time.time()

            forward_time.append(t1 - t0)
            backward_time.append(t2 - t1)

            if i % 20 == 0 and proc_id == 0:
                print("Step: {:05d}/{:05d}:{:05d} | Train Loss: {:.4f}".
                    format(i, len(loader), epoch, loss.item()))

        toc = time.time()
        if proc_id == 0:
            print("Epoch time: {:.05f}".format(toc-tic))
            print("Epoch {:05d} | Train Forward Time(s) {:.4f} | Backward Time(s) {:.4f}".
            format(epoch, forward_time[-1], backward_time[-1]))

        if n_gpus > 1:
            th.distributed.barrier()

    # sync for test
    if n_gpus > 1:
        th.distributed.barrier()

    if proc_id == 0:
        print("{}/{} Mean forward time: {:4f}".format(proc_id, n_gpus,
                                                      np.mean(forward_time[len(forward_time) // 4:])))
        print("{}/{} Mean backward time: {:4f}".format(proc_id, n_gpus,
                                                       np.mean(backward_time[len(backward_time) // 4:])))

def load_mag(args):
    dataset = DglNodePropPredDataset(name=args.dataset)
    split_idx = dataset.get_idx_split()
    train_idx = split_idx["train"]['paper']
    val_idx = split_idx["valid"]['paper']
    test_idx = split_idx["test"]['paper']
    hg_orig, labels = dataset[0]
    subgs = {}
    for etype in hg_orig.canonical_etypes:
        u, v = hg_orig.all_edges(etype=etype)
        subgs[etype] = (u, v)
        subgs[(etype[2], 'rev-'+etype[1], etype[0])] = (v, u)
    hg = dgl.heterograph(subgs)
    hg.nodes['paper'].data['feat'] = hg_orig.nodes['paper'].data['feat']
    labels = labels['paper'].squeeze()

    num_rels = len(hg.canonical_etypes)
    num_of_ntype = len(hg.ntypes)
    num_classes = dataset.num_classes
    print('Number of relations: {}'.format(num_rels))
    print('Number of class: {}'.format(num_classes))
    print('Number of train: {}'.format(len(train_idx)))
    print('Number of valid: {}'.format(len(val_idx)))
    print('Number of test: {}'.format(len(test_idx)))

    if args.node_feats:
        node_feats = []
        for ntype in hg.ntypes:
            if len(hg.nodes[ntype].data) == 0:
                node_feats.append(None)
            else:
                assert len(hg.nodes[ntype].data) == 1
                feat = hg.nodes[ntype].data.pop('feat')
                node_feats.append(feat.share_memory_())
    else:
        node_feats = [None] * num_of_ntype
    category = 'paper'
    return hg, node_feats, labels, train_idx, val_idx, test_idx, category, num_classes

def load_oag(args):
    dataset_file = args.dataset
    if not os.path.exists(dataset_file):
        dataset_url = "https://s3.us-west-2.amazonaws.com/dgl-data/dataset/OAG/" + dataset_file
        print("Fetching '{}'".format(dataset_url))
        subprocess.run(["wget", dataset_url]).check_returncode()

    dataset = dgl.load_graphs(dataset_file)[0]
    hg = dataset[0]

    # Construct node features.
    # TODO(zhengda) we need to construct the node features for author nodes.
    if args.node_feats:
        node_feats = []
        for ntype in hg.ntypes:
            if 'emb' in hg.nodes[ntype].data:
                feat = hg.nodes[ntype].data.pop('emb')
                node_feats.append(feat.share_memory_())
            else:
                node_feats.append(None)
    else:
        node_feats = [None] * len(hg.ntypes)

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

    print("Edge types:", len(hg.etypes))
    for etype in hg.canonical_etypes:
      print(etype)

    print("Number of nodes:", hg.num_nodes())
    print("Number of edges:", hg.num_edges())

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

def load_others(args):
    if args.dataset == 'aifb':
        dataset = AIFBDataset()
    elif args.dataset == 'mutag':
        dataset = MUTAGDataset()
    elif args.dataset == 'bgs':
        dataset = BGSDataset()
    elif args.dataset == 'am':
        dataset = AMDataset()
    else:
        raise ValueError()
    # Load from hetero-graph
    hg = dataset[0]

    num_rels = len(hg.canonical_etypes)
    num_of_ntype = len(hg.ntypes)
    category = dataset.predict_category
    num_classes = dataset.num_classes
    train_mask = hg.nodes[category].data.pop('train_mask')
    test_mask = hg.nodes[category].data.pop('test_mask')
    labels = hg.nodes[category].data.pop('labels')
    train_idx = th.nonzero(train_mask).squeeze()
    test_idx = th.nonzero(test_mask).squeeze()
    node_feats = [None] * num_of_ntype

    # AIFB, MUTAG, BGS and AM datasets do not provide validation set split.
    # Split train set into train and validation if args.validation is set
    # otherwise use train set as the validation set.
    if args.validation:
        val_idx = train_idx[:len(train_idx) // 5]
        train_idx = train_idx[len(train_idx) // 5:]
    else:
        val_idx = train_idx
    return hg, node_feats, labels, train_idx, val_idx, test_idx, category, num_classes


def main(args, devices):
    # load graph data
    if args.dataset == 'ogbn-mag':
        hg, node_feats, labels, train_idx, val_idx, test_idx, category, num_classes = load_mag(args)
    elif 'oag' in args.dataset:
        hg, node_feats, labels, train_idx, val_idx, test_idx, category, num_classes = load_oag(args)
    else:
        hg, node_feats, labels, train_idx, val_idx, test_idx, category, num_classes = load_others(args)

    # calculate norm for each edge type and store in edge
    if args.global_norm is False:
        for canonical_etype in hg.canonical_etypes:
            u, v, eid = hg.all_edges(form='all', etype=canonical_etype)
            _, inverse_index, count = th.unique(v, return_inverse=True, return_counts=True)
            degrees = count[inverse_index]
            norm = th.ones(eid.shape[0]) / degrees
            norm = norm.unsqueeze(1)
            hg.edges[canonical_etype].data['norm'] = norm

    # get target category id
    category_id = len(hg.ntypes)
    for i, ntype in enumerate(hg.ntypes):
        if ntype == category:
            category_id = i

    num_of_ntype = len(hg.ntypes)
    num_rels = len(hg.etypes)

    g = dgl.to_homogeneous(hg, edata=['norm'])
    if args.global_norm:
        u, v, eid = g.all_edges(form='all')
        _, inverse_index, count = th.unique(v, return_inverse=True, return_counts=True)
        degrees = count[inverse_index]
        norm = th.ones(eid.shape[0]) / degrees
        norm = norm.unsqueeze(1)
        g.edata['norm'] = norm

    g.ndata[dgl.NTYPE].share_memory_()
    g.edata[dgl.ETYPE].share_memory_()
    g.edata['norm'].share_memory_()
    node_ids = th.arange(g.number_of_nodes())

    # find out the target node ids
    node_tids = g.ndata[dgl.NTYPE]
    loc = (node_tids == category_id)
    target_idx = node_ids[loc]
    target_idx.share_memory_()
    train_idx.share_memory_()
    val_idx.share_memory_()
    test_idx.share_memory_()
    # Create csr/coo/csc formats before launching training processes with multi-gpu.
    # This avoids creating certain formats in each sub-process, which saves momory and CPU.
    g.create_formats_()

    n_gpus = len(devices)
    for i in range(len(devices)):
        if devices[i] == -1:
            devices[i] = 'cpu'

    if n_gpus == 1:
        run(0, n_gpus, args, devices,
            (g, node_feats, num_of_ntype, num_classes, num_rels, target_idx,
            train_idx, val_idx, test_idx, labels), None, None)
    # multi gpu
    else:
        queue = mp.Queue(n_gpus)
        procs = []
        num_train_seeds = train_idx.shape[0]
        num_valid_seeds = val_idx.shape[0]
        num_test_seeds = test_idx.shape[0]
        train_seeds = th.randperm(num_train_seeds)
        valid_seeds = th.randperm(num_valid_seeds)
        test_seeds = th.randperm(num_test_seeds)
        tseeds_per_proc = num_train_seeds // n_gpus
        vseeds_per_proc = num_valid_seeds // n_gpus
        tstseeds_per_proc = num_test_seeds // n_gpus
        for proc_id in range(n_gpus):
            # we have multi-gpu for training, evaluation and testing
            # so split trian set, valid set and test set into num-of-gpu parts.
            proc_train_seeds = train_seeds[proc_id * tseeds_per_proc :
                                           (proc_id + 1) * tseeds_per_proc \
                                           if (proc_id + 1) * tseeds_per_proc < num_train_seeds \
                                           else num_train_seeds]
            proc_valid_seeds = valid_seeds[proc_id * vseeds_per_proc :
                                           (proc_id + 1) * vseeds_per_proc \
                                           if (proc_id + 1) * vseeds_per_proc < num_valid_seeds \
                                           else num_valid_seeds]
            proc_test_seeds = test_seeds[proc_id * tstseeds_per_proc :
                                         (proc_id + 1) * tstseeds_per_proc \
                                         if (proc_id + 1) * tstseeds_per_proc < num_test_seeds \
                                         else num_test_seeds]
            p = mp.Process(target=run, args=(proc_id, n_gpus, args, devices,
                                             (g, node_feats, num_of_ntype, num_classes, num_rels, target_idx,
                                             train_idx, val_idx, test_idx, labels),
                                             (proc_train_seeds, proc_valid_seeds, proc_test_seeds),
                                             queue))
            p.start()
            procs.append(p)
        for p in procs:
            p.join()


def config():
    parser = argparse.ArgumentParser(description='RGCN')
    parser.add_argument("--dropout", type=float, default=0,
            help="dropout probability")
    parser.add_argument("--n-hidden", type=int, default=16,
            help="number of hidden units")
    parser.add_argument("--gpu", type=str, default='0',
            help="gpu")
    parser.add_argument("--lr", type=float, default=1e-2,
            help="learning rate")
    parser.add_argument("--n-bases", type=int, default=-1,
            help="number of filter weight matrices, default: -1 [use all]")
    parser.add_argument("--n-layers", type=int, default=2,
            help="number of propagation rounds")
    parser.add_argument("-e", "--n-epochs", type=int, default=50,
            help="number of training epochs")
    parser.add_argument("-d", "--dataset", type=str, required=True,
            help="dataset to use")
    parser.add_argument("--l2norm", type=float, default=0,
            help="l2 norm coef")
    parser.add_argument("--relabel", default=False, action='store_true',
            help="remove untouched nodes and relabel")
    parser.add_argument("--fanout", type=str, default="4, 4",
            help="Fan-out of neighbor sampling.")
    parser.add_argument("--use-self-loop", default=False, action='store_true',
            help="include self feature as a special relation")
    fp = parser.add_mutually_exclusive_group(required=False)
    fp.add_argument('--validation', dest='validation', action='store_true')
    fp.add_argument('--testing', dest='validation', action='store_false')
    parser.add_argument("--batch-size", type=int, default=100,
            help="Mini-batch size. ")
    parser.add_argument("--eval-batch-size", type=int, default=128,
            help="Mini-batch size. ")
    parser.add_argument("--num-workers", type=int, default=0,
            help="Number of workers for dataloader.")
    parser.add_argument("--low-mem", default=False, action='store_true',
            help="Whether use low mem RelGraphCov")
    parser.add_argument("--mix-cpu-gpu", default=False, action='store_true',
            help="Whether store node embeddins in cpu")
    parser.add_argument("--sparse-embedding", action='store_true',
            help='Use sparse embedding for node embeddings.')
    parser.add_argument('--node-feats', default=False, action='store_true',
            help='Whether use node features')
    parser.add_argument('--global-norm', default=False, action='store_true',
            help='User global norm instead of per node type norm')
    parser.add_argument('--layer-norm', default=False, action='store_true',
            help='Use layer norm')
    parser.set_defaults(validation=True)
    args = parser.parse_args()
    return args

if __name__ == '__main__':
    args = config()
    devices = list(map(int, args.gpu.split(',')))
    print(args)
    main(args, devices)
