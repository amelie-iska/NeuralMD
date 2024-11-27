import random
import torch
from torch_geometric.data import Data
from torch.utils.data import DataLoader


class BatchMISATO(Data):
    def __init__(self, batch=None, **kwargs):
        super(BatchMISATO, self).__init__(**kwargs)
        self.batch = batch

    @staticmethod
    def from_data_list(data_list):
        """Constructs a batch object from a python list holding
        :class:`torch_geometric.data.Data` objects.
        The assignment vector :obj:`batch` is created on the fly."""
        keys = [set(data.keys()) for data in data_list]
        keys = list(set.union(*keys))
        assert 'batch' not in keys

        batch = BatchMISATO()

        for key in keys:
            batch[key] = []
        batch.batch_ligand = []
        batch.batch_residue = []

        cumsum_node_ligand = 0
        cumsum_node_protein = 0

        for i, data in enumerate(data_list):
            num_nodes_ligand = data.ligand_x.size()[0]
            num_nodes_protein = data.protein_backbone_residue.size()[0]

            batch.batch_ligand.append(torch.full((num_nodes_ligand,), i, dtype=torch.long))
            batch.batch_residue.append(torch.full((num_nodes_protein,), i, dtype=torch.long))

            for key in data.keys():
                item = data[key]
                batch[key].append(item)

            cumsum_node_ligand += num_nodes_ligand
            cumsum_node_protein += num_nodes_protein

        for key in keys:
            batch[key] = torch.cat(batch[key], dim=data_list[0].__cat_dim__(key, batch[key][0]))
        batch.batch_ligand = torch.cat(batch.batch_ligand, dim=-1)
        batch.batch_residue = torch.cat(batch.batch_residue, dim=-1)
        return batch.contiguous()

    @property
    def num_graphs(self):
        """Returns the number of graphs in the batch."""
        return self.batch[-1].item() + 1


class DataLoaderMISATO(DataLoader):
    def __init__(self, dataset, batch_size=1, shuffle=True, **kwargs):
        super(DataLoaderMISATO, self).__init__(
            dataset,
            batch_size,
            shuffle,
            collate_fn=lambda data_list: BatchMISATO.from_data_list(data_list),
            **kwargs)