import torch.nn as nn
from dgl.nn.pytorch import GINEConv
from dgllife.model.gnn import MPNNGNN
import torch.nn.functional as F


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

        network_func = nn.Sequential(
            nn.Linear(edge_in_feats, node_out_feats))
        self.gnn_layer = GINEConv(apply_func=network_func,
                                  learn_eps=True)
        self.gnn_layer_remain = nn.ModuleList()
        network_func_remain = nn.Sequential(
            nn.Linear(node_out_feats, node_out_feats))
        for _ in range(num_step_message_passing - 1):
            self.gnn_layer_remain.append(GINEConv(apply_func=network_func,
                                                  learn_eps=True))
        self.project_node_feats_to_edge_feats = nn.Sequential(
            nn.Linear(node_in_feats, edge_in_feats),
            nn.ReLU())
        self.project_node_feats_to_node_out_feats = nn.Sequential(
            nn.Linear(edge_in_feats, node_out_feats),
            nn.ReLU())
        self.project_node_feats_to_node_feats = nn.Sequential(
            nn.Linear(node_out_feats, edge_in_feats),
            nn.ReLU())
        self.gru = nn.GRU(node_out_feats, node_out_feats)

    def forward(self, g, node_feats, edge_feats):
        node_feats = self.project_node_feats_to_edge_feats(node_feats)
        hidden_feats = self.project_node_feats_to_node_out_feats(node_feats).unsqueeze(0)

        node_feats = F.relu(self.gnn_layer(g, node_feats, edge_feats))
        node_feats, hidden_feats = self.gru(node_feats.unsqueeze(0), hidden_feats)
        node_feats = node_feats.squeeze(0)

        for remain_layer in self.gnn_layer_remain:
            node_feats = self.project_node_feats_to_node_feats(node_feats)
            node_feats = F.relu(remain_layer(g, node_feats, edge_feats))
            node_feats, hidden_feats = self.gru(node_feats.unsqueeze(0), hidden_feats)
            node_feats = node_feats.squeeze(0)
        return node_feats
