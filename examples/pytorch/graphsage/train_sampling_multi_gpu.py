import dgl
import numpy as np
import torch as th
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
import torch.multiprocessing as mp
from torch.cuda import nvtx
from torch.utils.data import DataLoader
from dgl.dataloading import AsyncTransferer
import dgl.function as fn
import dgl.nn.pytorch as dglnn
import time
import math
import argparse
from dgl.data import RedditDataset
from torch.nn.parallel import DistributedDataParallel
import tqdm
import traceback

from utils import thread_wrapped_func
from load_graph import load_ogb, load_reddit, inductive_split

class PrefetchingIterator:
    def __init__(self, i, dev_id, g):
        self.iter_ = i
        self.dev_id_ = dev_id
        self.g_ = g
        self.async_ = AsyncTransferer(dev_id)
        self.fetched_ = None

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
                input_nodes, seeds, blocks = self.get_next() 
                inputs, labels = load_subtensor(self.g_, self.g_.ndata['labels'],
                        seeds, input_nodes, self.dev_id_)
                blocks = [block.to(self.dev_id_) for block in blocks]
            finally:
                nvtx.range_pop()

        if self.dev_id_ != 'cpu':
            nvtx.range_push("prefetch")
            # initiate next fetch
            next_fetch = None
            try:
                input_nodes, seeds, next_blocks = self.get_next()
                next_inputs, next_labels, next_blocks = load_subtensor_future(
                    self.g_, next_blocks, self.g_.ndata['labels'], seeds,
                    input_nodes,  self.dev_id_, self.async_)
                next_fetch = (next_inputs, next_labels, next_blocks)
            except StopIteration:
                # nothing to fetch
                pass

            if self.fetched_:
                # grap futures
                inputs_future, labels_future, blocks = self.fetched_
                self.fetched_ = None
                inputs = inputs_future.wait()
                labels = labels_future
            self.fetched_ = next_fetch
            nvtx.range_pop()

        return (blocks, inputs, labels)
        

class PrefetchingNodeDataLoader:
    def __init__(self, dev_id, g, nids, block_sampler, **kwargs):
        self.dev_id_ = dev_id
        self.g_ = g
        self.dataloader_ = dgl.dataloading.NodeDataLoader(g, nids, block_sampler, **kwargs)

    def __iter__(self):
        """Return the iterator of the data loader."""
        nvtx.range_push("create_node_iterator")
        it = self.dataloader_.__iter__()
        nvtx.range_pop()
        nvtx.range_push("create_prefetching_iterator")
        ret = PrefetchingIterator(it, self.dev_id_, self.g_)
        nvtx.range_pop()
        return ret

    def __len__(self):
        """Return the number of batches of the data loader."""
        return len(self.dataloader_)


class SAGE(nn.Module):
    def __init__(self,
                 in_feats,
                 n_hidden,
                 n_classes,
                 n_layers,
                 activation,
                 dropout):
        super().__init__()
        self.n_layers = n_layers
        self.n_hidden = n_hidden
        self.n_classes = n_classes
        self.layers = nn.ModuleList()
        self.layers.append(dglnn.SAGEConv(in_feats, n_hidden, 'mean'))
        for i in range(1, n_layers - 1):
            self.layers.append(dglnn.SAGEConv(n_hidden, n_hidden, 'mean'))
        self.layers.append(dglnn.SAGEConv(n_hidden, n_classes, 'mean'))
        self.dropout = nn.Dropout(dropout)
        self.activation = activation

    def forward(self, blocks, x):
        h = x
        for l, (layer, block) in enumerate(zip(self.layers, blocks)):
            h = layer(block, h)
            if l != len(self.layers) - 1:
                h = self.activation(h)
                h = self.dropout(h)
        return h

    def inference(self, g, x, batch_size, device):
        """
        Inference with the GraphSAGE model on full neighbors (i.e. without neighbor sampling).
        g : the entire graph.
        x : the input of entire node set.

        The inference code is written in a fashion that it could handle any number of nodes and
        layers.
        """
        # During inference with sampling, multi-layer blocks are very inefficient because
        # lots of computations in the first few layers are repeated.
        # Therefore, we compute the representation of all nodes layer by layer.  The nodes
        # on each layer are of course splitted in batches.
        # TODO: can we standardize this?
        nodes = th.arange(g.number_of_nodes())
        for l, layer in enumerate(self.layers):
            y = th.zeros(g.number_of_nodes(), self.n_hidden if l != len(self.layers) - 1 else self.n_classes)

            sampler = dgl.dataloading.MultiLayerFullNeighborSampler(1)
            dataloader = dgl.dataloading.NodeDataLoader(
                g,
                th.arange(g.number_of_nodes()),
                sampler,
                batch_size=args.batch_size,
                shuffle=False,
                drop_last=False,
                num_workers=args.num_workers)

            for input_nodes, output_nodes, blocks in tqdm.tqdm(dataloader):
                block = blocks[0]

                block = block.to(device)
                h = x[input_nodes].to(device)
                h = layer(block, h)
                if l != len(self.layers) - 1:
                    h = self.activation(h)
                    h = self.dropout(h)

                y[output_nodes] = h.cpu()

            x = y
        return y

def compute_acc(pred, labels):
    """
    Compute the accuracy of prediction given the labels.
    """
    return (th.argmax(pred, dim=1) == labels).float().sum() / len(pred)

def evaluate(model, g, inputs, labels, val_nid, batch_size, device):
    """
    Evaluate the model on the validation set specified by ``val_nid``.
    g : The entire graph.
    inputs : The features of all the nodes.
    labels : The labels of all the nodes.
    val_nid : A node ID tensor indicating which nodes do we actually compute the accuracy for.
    batch_size : Number of nodes to compute at the same time.
    device : The GPU device to evaluate on.
    """
    model.eval()
    with th.no_grad():
        pred = model.inference(g, inputs, batch_size, device)
    model.train()
    return compute_acc(pred[val_nid], labels[val_nid])

def load_subtensor(g, labels, seeds, input_nodes, dev_id):
    """
    Copys features and labels of a set of nodes onto GPU.
    """
    nvtx.range_push("batch_inputs_to_gpu")
    if dev_id != 'cpu':
        batch_inputs_cpu = th.empty((input_nodes.shape[0], g.ndata['features'].shape[1]), pin_memory=True)
        th.index_select(g.ndata['features'],0, input_nodes, out=batch_inputs_cpu)
        batch_inputs = batch_inputs_cpu.to(dev_id, non_blocking=True)
    else:
        batch_inputs = g.ndata['features'][input_nodes].to(dev_id)
    nvtx.range_pop()
    nvtx.range_push("batch_labels_to_gpu")
    batch_labels = labels[seeds].to(dev_id, non_blocking=True).long()
    nvtx.range_pop()
    return batch_inputs, batch_labels


def load_subtensor_future(g, blocks, labels, seeds, input_nodes, dev_id, transfer):
    """
    Copys features and labels of a set of nodes onto GPU.
    """
    nvtx.range_push("batch_labels_to_gpu_async")
    batch_labels = labels[seeds].to(dev_id, non_blocking=True).long()
    nvtx.range_pop()

    nvtx.range_push("batch_inputs_to_gpu_async")
    batch_inputs_cpu = th.empty((input_nodes.shape[0], g.ndata['features'].shape[1]), pin_memory=True)
    th.index_select(g.ndata['features'],0, input_nodes, out=batch_inputs_cpu)
    batch_blocks = [block.to(dev_id, non_blocking=True) for block in blocks]
    batch_inputs = transfer.async_copy(batch_inputs_cpu, dev_id)
    nvtx.range_pop()
    return batch_inputs, batch_labels, batch_blocks

#### Entry point

def run(proc_id, n_gpus, args, devices, data):
    # Start up distributed training, if enabled.
    dev_id = devices[proc_id]
    if n_gpus > 1:
        dist_init_method = 'tcp://{master_ip}:{master_port}'.format(
            master_ip='127.0.0.1', master_port='12345')
        world_size = n_gpus
        backend = "nccl"
        if dev_id == 'cpu':
            backend = "gloo"
        print("Using backend '{}' for '{}'".format(backend, dev_id))
        th.distributed.init_process_group(backend=backend,
                                          init_method=dist_init_method,
                                          world_size=world_size,
                                          rank=proc_id)
    if dev_id != 'cpu':
        th.cuda.set_device(dev_id)

    # Unpack data
    in_feats, n_classes, train_g, val_g, test_g = data
    train_mask = train_g.ndata['train_mask']
    val_mask = val_g.ndata['val_mask']
    test_mask = ~(test_g.ndata['train_mask'] | test_g.ndata['val_mask'])
    train_nid = train_mask.nonzero().squeeze()
    val_nid = val_mask.nonzero().squeeze()
    test_nid = test_mask.nonzero().squeeze()

    # Split train_nid
    train_nid = th.split(train_nid, math.ceil(len(train_nid) / n_gpus))[proc_id]

    # Create PyTorch DataLoader for constructing blocks
    sampler = dgl.dataloading.MultiLayerNeighborSampler(
        [int(fanout) for fanout in args.fan_out.split(',')])
    dataloader = PrefetchingNodeDataLoader(
        dev_id,
        train_g,
        train_nid,
        sampler,
        batch_size=args.batch_size,
        shuffle=True,
        drop_last=False,
        num_workers=args.num_workers,
        persistent_workers=True)

    # Define model and optimizer
    model = SAGE(in_feats, args.num_hidden, n_classes, args.num_layers, F.relu, args.dropout)
    model = model.to(dev_id)
    if n_gpus > 1:
        if dev_id != 'cpu':
            model = DistributedDataParallel(model, device_ids=[dev_id], output_device=dev_id)
        else:
            model = DistributedDataParallel(model, device_ids=None, output_device=None)
    loss_fcn = nn.CrossEntropyLoss()
    loss_fcn = loss_fcn.to(dev_id)
    optimizer = optim.Adam(model.parameters(), lr=args.lr)

    # Training loop
    avg = 0
    iter_tput = []
    fwd_tput = []
    bwd_tput = []
    for epoch in range(args.num_epochs):
        tic = time.time()

        # Loop over the dataloader to sample the computation dependency graph as a list of
        # blocks.
        nvtx.range_push("epoch_{}".format(epoch))
        for step, (blocks, batch_inputs, batch_labels) in enumerate(dataloader):
            tic_step = time.time()

            count = batch_labels.shape[0]

            # Compute loss and prediction
            nvtx.range_push("model.forward")
            batch_pred = model(blocks, batch_inputs)
            nvtx.range_pop()

            toc_step = time.time()

            nvtx.range_push("loss_fcn")
            loss = loss_fcn(batch_pred, batch_labels)
            nvtx.range_pop()

            nvtx.range_push("optimizer.zero_grad")
            optimizer.zero_grad()
            nvtx.range_pop()

            nvtx.range_push("loss.backward")
            loss.backward()
            nvtx.range_pop()

            nvtx.range_push("optimizer.step")
            optimizer.step()
            nvtx.range_pop()

            batch_inputs = None

            nvtx.range_push("reporting.iter_tput")
            if proc_id == 0:
                iter_tput.append(count * n_gpus / (time.time() - tic_step))
                fwd_tput.append(toc_step-tic_step)
                bwd_tput.append(time.time()-toc_step)
            nvtx.range_pop()

            nvtx.range_push("reporting.step")
            if step % args.log_every == 0 and proc_id == 0:
                acc = compute_acc(batch_pred, batch_labels)
                print('Epoch {:05d} | Step {:05d} | Loss {:.4f} | Train Acc {:.4f} | Speed (samples/sec) {:.4f} | FWD: {:.4f}s | BWD {:.4f}s | GPU {:.1f} MB'.format(
                    epoch, step, loss.item(), acc.item(), np.mean(iter_tput[3:]), np.mean(fwd_tput[3:]),np.mean(bwd_tput[3:]), th.cuda.max_memory_allocated() / 1000000))
            nvtx.range_pop()

        nvtx.range_push("sync")
        if n_gpus > 1:
            th.distributed.barrier()
        nvtx.range_pop()
        nvtx.range_pop()

        nvtx.range_push("epoch_reporting")
        toc = time.time()
        if proc_id == 0:
            print('Epoch Time(s): {:.4f}'.format(toc - tic))
            if epoch >= 5:
                avg += toc - tic
            if epoch % args.eval_every == 0 and epoch != 0:
                if n_gpus == 1:
                    eval_acc = evaluate(
                        model, val_g, val_g.ndata['features'], val_g.ndata['labels'], val_nid, args.batch_size, devices[0])
                    test_acc = evaluate(
                        model, test_g, test_g.ndata['features'], test_g.ndata['labels'], test_nid, args.batch_size, devices[0])
                else:
                    eval_acc = evaluate(
                        model.module, val_g, val_g.ndata['features'], val_g.ndata['labels'], val_nid, args.batch_size, devices[0])
                    test_acc = evaluate(
                        model.module, test_g, test_g.ndata['features'], test_g.ndata['labels'], test_nid, args.batch_size, devices[0])
                print('Eval Acc {:.4f}'.format(eval_acc))
                print('Test Acc: {:.4f}'.format(test_acc))

        nvtx.range_pop()

    if n_gpus > 1:
        th.distributed.barrier()
    if proc_id == 0:
        print('Avg epoch time: {}'.format(avg / (epoch - 4)))

if __name__ == '__main__':
    argparser = argparse.ArgumentParser("multi-gpu training")
    argparser.add_argument('--gpu', type=str, default='0',
        help="Comma separated list of GPU device IDs.")
    argparser.add_argument('--num-epochs', type=int, default=20)
    argparser.add_argument('--num-hidden', type=int, default=16)
    argparser.add_argument('--num-layers', type=int, default=2)
    argparser.add_argument('--fan-out', type=str, default='10,25')
    argparser.add_argument('--batch-size', type=int, default=1000)
    argparser.add_argument('--log-every', type=int, default=20)
    argparser.add_argument('--eval-every', type=int, default=5)
    argparser.add_argument('--lr', type=float, default=0.003)
    argparser.add_argument('--dataset', type=str, default='ogbn-products')
    argparser.add_argument('--dropout', type=float, default=0.5)
    argparser.add_argument('--num-workers', type=int, default=0,
        help="Number of sampling processes. Use 0 for no extra process.")
    argparser.add_argument('--inductive', action='store_true',
        help="Inductive learning setting")
    args = argparser.parse_args()
    
    devices = list(map(int, args.gpu.split(',')))
    n_gpus = len(devices)

    if args.dataset == 'reddit':
        g, n_classes = load_reddit()
    else:
        g, n_classes = load_ogb(args.dataset)

    # Construct graph
    in_feats = g.ndata['features'].shape[1]

    print("Splitting...")
    if args.inductive:
        train_g, val_g, test_g = inductive_split(g)
    else:
        train_g = val_g = test_g = g

    # Create csr/coo/csc formats before launching training processes with multi-gpu.
    # This avoids creating certain formats in each sub-process, which saves momory and CPU.
    print("Creating formats...")
    formats_start = time.time()
    train_g.create_formats_()
    val_g.create_formats_()
    test_g.create_formats_()
    print("Creating formats took {:.05f}s".format(time.time()-formats_start))
    # Pack data
    data = in_feats, n_classes, train_g, val_g, test_g

    for i in range(len(devices)):
        if devices[i] < 0:
            devices[i] = 'cpu'

    print("Launching run...")
    if n_gpus == 1:
        run(0, n_gpus, args, devices, data)
    else:
        procs = []
        for proc_id in range(n_gpus):
            p = mp.Process(target=thread_wrapped_func(run),
                           args=(proc_id, n_gpus, args, devices, data))
            p.start()
            procs.append(p)
        for p in procs:
            p.join()
