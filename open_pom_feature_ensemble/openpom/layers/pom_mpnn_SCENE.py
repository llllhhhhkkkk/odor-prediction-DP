import dgl
import torch
import torch.nn as nn
from dgl.nn.pytorch import EdgeGATConv, SetTransformerEncoder
from dgllife.model.gnn import MPNNGNN
import torch.nn.functional as F


class CustomMPNNGNN(MPNNGNN):
    def __init__(self,
                 layer_dims,
                 node_in_feats: int = 134,
                 edge_in_feats: int = 6,
                 node_out_feats: int = 64,
                 edge_hidden_feats: int = 128,
                 num_step_message_passing: int = 1,
                 residual: bool = True,
                 num_heads: int = 1,
                 message_aggregator_type: str = 'sum'):

        super(CustomMPNNGNN,
              self).__init__(node_in_feats=node_in_feats,
                             edge_in_feats=edge_in_feats,
                             node_out_feats=node_out_feats,
                             edge_hidden_feats=edge_hidden_feats,
                             num_step_message_passing=num_step_message_passing)
        if layer_dims is None:
            layer_dims = [16, 32, 64]
        self.project_node_feats = nn.Sequential(
            nn.Linear(node_in_feats, node_out_feats),
            nn.ReLU()
        )
        self.layer_dims = layer_dims
        self.GNN_layers = nn.ModuleList()
        self.LayerNorms = nn.ModuleList()
        self.TopKPooling = nn.ModuleList()
        layer_dims_len = len(layer_dims)
        for i in range(layer_dims_len - 1):
            self.GNN_layers.append(
                EdgeGATConv(
                    in_feats=layer_dims[i],
                    edge_feats=edge_in_feats,
                    out_feats=layer_dims[i+1] // num_heads,  # 输出特征维度需要除以头数
                    num_heads=num_heads,
                    feat_drop=0.01,
                    attn_drop=0.01,
                    negative_slope=0.2,
                    residual=residual,
                    allow_zero_in_degree=True,
                    activation=F.relu,
                    bias=True
                )
            )
            self.LayerNorms.append(nn.LayerNorm(layer_dims[i+1]))
            self.TopKPooling.append(TopKPooling(layer_dims[i+1], 0.9, num_heads))
        self.project_node_feats_temp = nn.Sequential(
            nn.Linear(node_in_feats, self.layer_dims[0]),
            nn.ReLU()
        )
        self.set_trans_enc = SetTransformerEncoder(134, 4, 4, 20)

    def reset_parameters(self):
        """Reinitialize model parameters."""
        self.project_node_feats[0].reset_parameters()
        self.gnn_layer.reset_parameters()
        for layer in self.gnn_layer.edge_func:
            if isinstance(layer, nn.Linear):
                layer.reset_parameters()
        for layer in self.GNN_layers:
            layer.reset_parameters()
        self.gru.reset_parameters()

    def forward(self, g, node_feats, edge_feats):
        node_feats = self.set_trans_enc(g, node_feats)
        temp_node_feats = node_feats
        temp_node_feats = self.project_node_feats(temp_node_feats)  # (V, node_out_feats)
        hidden_feats = temp_node_feats.unsqueeze(0)  # (1, V, node_out_feats)
        node_feats = self.project_node_feats_temp(node_feats)
        g = dgl.add_self_loop(g)
        for i in range(g.num_edges() - g.num_nodes(), g.num_edges()):
            g.edata['edge_attr'][i] = torch.zeros(6)
        edge_feats = g.edata['edge_attr']
        for i, gnn_layer in enumerate(self.GNN_layers):
            node_feats = gnn_layer(g, node_feats, edge_feats).flatten(1)
            node_feats = self.LayerNorms[i](node_feats)
        node_feats, hidden_feats = self.gru(node_feats.unsqueeze(0), hidden_feats)
        node_feats = node_feats.squeeze(0)

        return node_feats


class TopKPooling(nn.Module):
    def __init__(self, in_feats, ratio=0.9, num_heads=1):
        super(TopKPooling, self).__init__()
        self.in_feats = in_feats
        self.ratio = ratio
        self.num_heads = num_heads
        self.attention = nn.MultiheadAttention(in_feats, num_heads)

    def forward(self, g, feat):
        # Calculate self-attention
        attn_output, _ = self.attention(feat, feat, feat)

        # Calculate attention scores
        scores = attn_output.mean(dim=-1)

        k = int(g.number_of_nodes() * self.ratio)
        _, indices = torch.topk(scores, k, dim=0)

        g = dgl.node_subgraph(g, indices)
        return g, feat[indices], indices