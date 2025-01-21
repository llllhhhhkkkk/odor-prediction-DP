import torch
from dgl.nn.pytorch import GMMConv
from dgllife.model.gnn import MPNNGNN
import torch.nn.functional as F
import dgl
import torch.nn as nn


class CustomMPNNGNN(MPNNGNN):
    """
    Customized MPNNGNN layer based MPNNGNN layer in dgllife library.

    Additional options:
    -> toggle for residual in gnn layer
    -> choice for message aggregator type

    MPNN is introduced in `Neural Message Passing for Quantum Chemistry
    <https://arxiv.org/abs/1704.01212>`__.

    This class performs message passing in MPNN
    and returns the updated node representations.
    """

    def __init__(self,
                 node_in_feats: int = 50,
                 edge_in_feats: int = 50,
                 node_out_feats: int = 64,
                 edge_hidden_feats: int = 128,
                 num_step_message_passing: int = 6,
                 residual: bool = True,
                 message_aggregator_type: str = 'sum'):
        """
        Parameters
        ----------
        node_in_feats: int
            Size for the input node features.
        node_out_feats: int
            Size for the output node representations. Default to 64.
        edge_in_feats: int
            Size for the input edge features. Default to 128.
        edge_hidden_feats: int
            Size for the hidden edge representations.
        num_step_message_passing: int
            Number of message passing steps. Default to 6.
        residual: bool
            If true, adds residual layer to gnn layer
        message_aggregator_type: str
            message aggregator type, 'sum', 'mean' or 'max'
        """
        super(CustomMPNNGNN,
              self).__init__(node_in_feats=node_in_feats,
                             edge_in_feats=edge_in_feats,
                             node_out_feats=node_out_feats,
                             edge_hidden_feats=edge_hidden_feats,
                             num_step_message_passing=num_step_message_passing)

        self.gnn_layer = GMMConv(in_feats=node_out_feats,
                                 out_feats=node_out_feats,
                                 dim=6,
                                 residual=residual,
                                 n_kernels=2,
                                 aggregator_type=message_aggregator_type,
                                 )

        self.layer_norm = nn.LayerNorm(node_out_feats)

    def forward(self, g, node_feats, edge_feats):
        node_feats = self.project_node_feats(node_feats)  # (V, node_out_feats)
        hidden_feats = node_feats.unsqueeze(0)  # (1, V, node_out_feats)
        g = dgl.add_self_loop(g)
        for i in range(g.num_edges() - g.num_nodes(), g.num_edges()):
            g.edata['edge_attr'][i] = torch.zeros(6)
        edge_feats = g.edata['edge_attr']
        for _ in range(self.num_step_message_passing):
            node_feats = self.layer_norm(F.leaky_relu(self.gnn_layer(g, node_feats, edge_feats)))
            node_feats, hidden_feats = self.gru(node_feats.unsqueeze(0), hidden_feats)
            node_feats = node_feats.squeeze(0)

        return node_feats
