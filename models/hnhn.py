import torch
import torch.nn as nn
from torch_geometric.data import Batch
from torch_geometric.nn import MLP, global_add_pool
from ogb.graphproppred.mol_encoder import AtomEncoder
from dhg.nn import HNHNConv

from models.utils import data2hg


class HNHN(nn.Module):
    r"""The HNHN model proposed in `HNHN: Hypergraph Networks with Hyperedge Neurons <https://arxiv.org/pdf/2006.12278.pdf>`_ paper (ICML 2020).

    Args:
        ``in_channels`` (``int``): :math:`C_{in}` is the number of input channels.
        ``hid_channels`` (``int``): :math:`C_{hid}` is the number of hidden channels.
        ``num_classes`` (``int``): The Number of class of the classification task.
        ``use_bn`` (``bool``): If set to ``True``, use batch normalization. Defaults to ``False``.
        ``drop_rate`` (``float``, optional): Dropout ratio. Defaults to ``0.5``.
    """

    def __init__(
        self,
        hid_channels: int,
        num_layer: int,
        use_bn: bool = False,
        drop_rate: float = 0.5,
    ) -> None:
        super().__init__()
        self.atom_encoder = AtomEncoder(emb_dim=hid_channels)
        self.layers = nn.ModuleList()
        for _ in range(num_layer - 1):
            self.layers.append(
                HNHNConv(hid_channels, hid_channels, use_bn=use_bn, drop_rate=drop_rate)
            )
        self.layers.append(
            HNHNConv(hid_channels, hid_channels, use_bn=use_bn, is_last=True)
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
        for layer in self.layers:
            x = layer(x, hg)
        x = global_add_pool(x, data.batch)
        x = self.mlp_out(x)
        return x.view(-1)
