import torch
import torch.nn as nn
from torch_geometric.data import Batch
from torch_geometric.nn import MLP, global_add_pool
from ogb.graphproppred.mol_encoder import AtomEncoder
from dhg.nn import HyperGCNConv
from dhg.structure.graphs import Graph

from models.utils import data2hg


class HyperGCN(nn.Module):
    r"""The HyperGCN model proposed in `HyperGCN: A New Method of Training Graph Convolutional Networks on Hypergraphs <https://papers.nips.cc/paper/2019/file/1efa39bcaec6f3900149160693694536-Paper.pdf>`_ paper (NeurIPS 2019).
    
    Args:
        ``in_channels`` (``int``): :math:`C_{in}` is the number of input channels.
        ``hid_channels`` (``int``): :math:`C_{hid}` is the number of hidden channels.
        ``num_classes`` (``int``): The Number of class of the classification task.
        ``use_mediator`` (``str``): Whether to use mediator to transform the hyperedges to edges in the graph. Defaults to ``False``.
        ``fast`` (``bool``): If set to ``True``, the transformed graph structure will be computed once from the input hypergraph and vertex features, and cached for future use. Defaults to ``True``.
        ``drop_rate`` (``float``, optional): Dropout ratio. Defaults to 0.5.
    """

    def __init__(
        self,
        hid_channels: int,
        num_layer: int,
        use_mediator: bool = False,
        use_bn: bool = False,
        fast: bool = True,
        drop_rate: float = 0.5,
    ) -> None:
        super().__init__()
        self.fast = fast
        self.cached_g = None
        self.with_mediator = use_mediator
        self.atom_encoder = AtomEncoder(emb_dim=hid_channels)
        self.layers = nn.ModuleList()
        for _ in range(num_layer - 1):
            self.layers.append(
                HyperGCNConv(
                    hid_channels, hid_channels, use_mediator, use_bn=use_bn, drop_rate=drop_rate,
                )
            )
        self.layers.append(
            HyperGCNConv(
                hid_channels, hid_channels, use_mediator, use_bn=use_bn, is_last=True
            )
        )
        self.mlp_out = MLP(
            in_channels=hid_channels, hidden_channels=hid_channels, out_channels=1, num_layers=3
        )

    def forward(self, data: Batch) -> torch.Tensor:
        r"""The forward function.

        Args:
            ``X`` (``torch.Tensor``): Input vertex feature matrix. Size :math:`(N, C_{in})`.
            ``hg`` (``dhg.Hypergraph``): The hypergraph structure that contains :math:`N` vertices.
        """
        x = self.atom_encoder(data.x)
        hg = data2hg(data)
        if self.fast:
            if self.cached_g is None:
                self.cached_g = Graph.from_hypergraph_hypergcn(
                    hg, x, self.with_mediator
                )
            for layer in self.layers:
                x = layer(x, hg, self.cached_g)
        else:
            for layer in self.layers:
                x = layer(x, hg)
        x = global_add_pool(x, data.batch)
        x = self.mlp_out(x)
        return x.view(-1)
