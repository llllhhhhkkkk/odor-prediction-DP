from dgl.nn.pytorch import PNAConv
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
                 message_aggregator_type=None,
                 scalers=None,
                 delta=2.5,
                 num_towers=1):
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

        if message_aggregator_type is None:
            message_aggregator_type = ['sum']
        if scalers is None:
            scalers = ['identity', 'amplification']
        self.gnn_layer = PNAConv(in_size=node_out_feats,
                                 out_size=node_out_feats,
                                 edge_feat_size=edge_in_feats,
                                 num_towers=num_towers,
                                 aggregators=message_aggregator_type,
                                 scalers=scalers,
                                 delta=delta,
                                 residual=residual)

    def forward(self, g, node_feats, edge_feats):
        node_feats = self.project_node_feats(node_feats) # (V, node_out_feats)
        hidden_feats = node_feats.unsqueeze(0)           # (1, V, node_out_feats)

        for _ in range(self.num_step_message_passing):
            node_feats = F.relu(self.gnn_layer(g, node_feats, edge_feats))
            node_feats, hidden_feats = self.gru(node_feats.unsqueeze(0), hidden_feats)
            node_feats = node_feats.squeeze(0)

        return node_feats
