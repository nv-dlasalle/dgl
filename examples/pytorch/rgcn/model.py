import torch as th
import torch.nn as nn
from torch.cuda import nvtx

class BaseRGCN(nn.Module):
    def __init__(self, num_nodes, h_dim, out_dim, num_rels, num_bases,
                 num_hidden_layers=1, dropout=0,
                 use_self_loop=False, use_cuda=False):
        super(BaseRGCN, self).__init__()
        self.num_nodes = num_nodes
        self.h_dim = h_dim
        self.out_dim = out_dim
        self.num_rels = num_rels
        self.num_bases = None if num_bases < 0 else num_bases
        self.num_hidden_layers = num_hidden_layers
        self.dropout = dropout
        self.use_self_loop = use_self_loop
        self.use_cuda = use_cuda

        # create rgcn layers
        self.build_model()

    def build_model(self):
        self.layers = nn.ModuleList()
        # i2h
        i2h = self.build_input_layer()
        if i2h is not None:
            self.layers.append(i2h)
        # h2h
        for idx in range(self.num_hidden_layers):
            h2h = self.build_hidden_layer(idx)
            self.layers.append(h2h)
        # h2o
        h2o = self.build_output_layer()
        if h2o is not None:
            self.layers.append(h2o)

    def build_input_layer(self):
        return None

    def build_hidden_layer(self, idx):
        raise NotImplementedError

    def build_output_layer(self):
        return None

    def forward(self, g, h, r, norm):
        for layer in self.layers:
            h = layer(g, h, r, norm)
        return h

class RelGraphEmbedLayer(nn.Module):
    r"""Embedding layer for featureless heterograph.
    Parameters
    ----------
    dev_id : int
        Device to run the layer.
    num_nodes : int
        Number of nodes.
    node_tides : tensor
        Storing the node type id for each node starting from 0
    num_of_ntype : int
        Number of node types
    input_size : list of int
        A list of input feature size for each node type. If None, we then
        treat certain input feature as an one-hot encoding feature.
    embed_size : int
        Output embed size
    embed_name : str, optional
        Embed name
    """
    def __init__(self,
                 dev_id,
                 num_nodes,
                 node_tids,
                 num_of_ntype,
                 input_size,
                 embed_size,
                 sparse_emb=False,
                 embed_name='embed'):
        super(RelGraphEmbedLayer, self).__init__()
        self.dev_id = th.device(dev_id if dev_id >= 0 else 'cpu')
        self.embed_size = embed_size
        self.embed_name = embed_name
        self.num_nodes = num_nodes
        self.sparse_emb = sparse_emb

        # create weight embeddings for each node for each relation
        self.embeds = nn.ParameterDict()
        self.num_of_ntype = num_of_ntype
        self.idmap = th.empty(num_nodes).long()

        for ntype in range(num_of_ntype):
            if input_size[ntype] is not None:
                input_emb_size = input_size[ntype].shape[1]
                embed = nn.Parameter(th.empty(input_emb_size, self.embed_size, device=self.dev_id))
                nn.init.xavier_uniform_(embed)
                self.embeds[str(ntype)] = embed

        self.node_embeds = th.nn.Embedding(node_tids.shape[0], self.embed_size, sparse=self.sparse_emb)
        nn.init.uniform_(self.node_embeds.weight, -1.0, 1.0)

    def forward(self, node_ids, node_tids, type_ids, features, loc_cpu):
        """Forward computation
        Parameters
        ----------
        node_ids : tensor
            node ids to generate embedding for.
        node_ids : tensor
            node type ids
        features : list of features
            list of initial features for nodes belong to different node type.
            If None, the corresponding features is an one-hot encoding feature,
            else use the features directly as input feature and matmul a
            projection matrix.
        Returns
        -------
        tensor
            embeddings as the input of the next layer
        """
        nvtx.range_push("node_ids_to_gpu")
        tsd_ids = node_ids.to(self.node_embeds.weight.device)
        nvtx.range_pop()
        nvtx.range_push("alloc_empty_embds")
        embeds = th.empty(node_ids.shape[0], self.embed_size, device=self.dev_id)
        nvtx.range_pop()

        nvtx.range_push("bulk_embed")
        locs = []
        locs_prefix=[0]
        for ntype in range(self.num_of_ntype):
            if features[ntype] is None:
                nvtx.range_push("generate_loc")
                locs.append(loc_cpu[ntype])
                locs_prefix.append(locs_prefix[-1]+locs[-1].shape[0])
                nvtx.range_pop()

        all_embed_locs = tsd_ids[th.cat(locs, 0)]
        all_embed = self.node_embeds(all_embed_locs).pin_memory().to(self.dev_id, non_blocking=True)
        nvtx.range_pop()

        for ntype in range(self.num_of_ntype):
            if features[ntype] is not None:
                nvtx.range_push("generate_loc")
                loc = loc_cpu[ntype] 
                loc_gpu = loc.to(self.dev_id, non_blocking=True)
                nvtx.range_pop()
                nvtx.range_push("embed_with_features")
                embeds[loc_gpu] = features[ntype] @ self.embeds[str(ntype)]
                nvtx.range_pop()

        i = 0
        for ntype in range(self.num_of_ntype):
            if features[ntype] is None:
                nvtx.range_push("generate_loc")
                loc = loc_cpu[ntype] 
                loc_gpu = loc.to(self.dev_id, non_blocking=True)
                nvtx.range_pop()
                nvtx.range_push("embed_without_features")
                embeds[loc_gpu] = all_embed[locs_prefix[i]:locs_prefix[i+1]]
                i += 1
                nvtx.range_pop()

        return embeds
