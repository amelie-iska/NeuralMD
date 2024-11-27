import math
from math import pi as PI

import torch
from torch import nn
from torch import Tensor

from torch_geometric.nn import radius_graph, radius
from torch_geometric.utils import add_self_loops, remove_self_loops
from torch_scatter import scatter
from .FrameNet_support import _normalize, MLP, RBF_repredding_01, RBF_repredding_02, weight_initialization, kaiming_uniform


class ComplexFrameLayer(nn.Module):
    def __init__(
        self, latent_dim, cutoff,
        batch_norm, momentum, dropout, readout,
    ):
        super(ComplexFrameLayer, self).__init__()
        self.latent_dim = latent_dim
        self.cutoff = cutoff
        self.readout = readout

        self.input_layer = nn.Sequential(
            nn.Linear(self.latent_dim, self.latent_dim),
            nn.SiLU(inplace=True),
            nn.Linear(self.latent_dim, self.latent_dim),
        )
        
        self.scalarization_linear_01 = nn.Sequential(
            nn.Linear(3, self.latent_dim // 4),
            nn.SiLU(inplace=True),
            nn.Linear(self.latent_dim // 4, 1))

        self.scalarization_linear_02 = nn.Sequential(
            nn.Linear(3, self.latent_dim // 4),
            nn.SiLU(inplace=True),
            nn.Linear(self.latent_dim // 4, 1))
        return

    def forward(self,
        h_ligand, pos_ligand, batch_ligand, edge_ligand_idx, ligand_vec,
        h_residue, pos_residue, batch_residue, edge_residue_idx,
        edge_diff, residue_frame):

        # [num_edge, latent_dim]
        h_edge_invariant = h_ligand[edge_ligand_idx] + h_residue[edge_residue_idx]
        h_edge_invariant = self.input_layer(h_edge_invariant)

        # [num_edge, 3, latent_dim]
        h_edge_equivariant = h_edge_invariant.unsqueeze(1) * edge_diff.unsqueeze(2)

        # [num_edge, 3, 3, latent_dim]
        h_edge_scalarization = h_edge_equivariant.unsqueeze(2) * residue_frame[edge_residue_idx].unsqueeze(-1)
        # [num_edge, 3, latent_dim]
        h_edge_scalarization = torch.sum(h_edge_scalarization, dim=1)
        # [num_edge, latent_dim]
        h_edge_weight_01 = (self.scalarization_linear_01(torch.permute(h_edge_scalarization, (0, 2, 1))) + torch.permute(h_edge_scalarization, (0, 2, 1))[:, :,0].unsqueeze(2)).squeeze(-1)
        h_edge_weight_02 = (self.scalarization_linear_02(torch.permute(h_edge_scalarization, (0, 2, 1))) + torch.permute(h_edge_scalarization, (0, 2, 1))[:, :,0].unsqueeze(2)).squeeze(-1)

        # [num_edge, 3, latent_dim]
        ligand_vec_expanded = ligand_vec[edge_ligand_idx] * h_edge_weight_01.unsqueeze(1) + edge_diff.unsqueeze(2) * h_edge_weight_02.unsqueeze(1)
        dim_size = pos_ligand.shape[0]
        # [num_ligand, 3, latent_dim]
        ligand_vec_gathered = scatter(ligand_vec_expanded, edge_ligand_idx, dim=0, dim_size=dim_size)
        return ligand_vec_gathered


class FrameNetComplex01(nn.Module):
    def __init__(
        self,
        latent_dim, num_layer, cutoff=8,
        num_radial=32, rbf_type="RBF_repredding_01", rbf_gamma=None,
        readout="mean",
        batch_norm=True, momentum=0.2, dropout=0.2,
    ):
        super(FrameNetComplex01, self).__init__()
        self.latent_dim = latent_dim
        self.num_layer = num_layer
        self.cutoff = cutoff
        self.readout = readout

        self.complex_frame_layers = nn.ModuleList()

        for _ in range(self.num_layer):
            self.complex_frame_layers.append(
                ComplexFrameLayer(
                    latent_dim=latent_dim, cutoff=self.cutoff,
                    batch_norm=batch_norm, momentum=momentum, dropout=dropout, readout=readout)
            )

        self.last_layers = MLP(
            [self.latent_dim, self.latent_dim, 1], batch_norm=False, dropout=dropout, momentum=momentum)
        return
        
    def forward(self, ligand_repr, ligand_vec_input, pos_ligand, batch_ligand, residue_repr, pos_residue, batch_residue):
        num_residue = residue_repr.size()[0]

        # [num_residue, latent_dim]
        num_node = residue_repr.size()[0]
        sequence_id_list = torch.arange(num_node, device=pos_residue.device)

        u = torch.nn.functional.normalize(pos_residue[1:,:] - pos_residue[:-1,:], dim=1)
        start = u[1:,:]
        end = u[:-1, :]
        diff = torch.nn.functional.normalize(end - start, dim=1)
        cross = torch.nn.functional.normalize(torch.cross(end, start, dim=1), dim=1)
        vertical = torch.nn.functional.normalize(torch.cross(diff, cross, dim=1), dim=1)
        frame = torch.stack([diff, cross, vertical], dim=1)
        residue_frame = torch.cat([frame[0].unsqueeze(0), frame, frame[-1].unsqueeze(0)], dim=0)

        edge_residue_idx, edge_ligand_idx = radius(pos_ligand, pos_residue, self.cutoff, batch_ligand, batch_residue)

        edge_diff = pos_ligand[edge_ligand_idx] - pos_residue[edge_residue_idx]
        edge_diff = _normalize(edge_diff)
        
        h_ligand = ligand_repr
        h_residue = residue_repr
        # [num_ligand, 3, latent_dim]
        ligand_vec = torch.zeros(h_ligand.size(0), 3, h_ligand.size(1), device=h_ligand.device)
        for layer_idx, layer in enumerate(self.complex_frame_layers):
            dligand_vec = layer(
                h_ligand=h_ligand, pos_ligand=pos_ligand, batch_ligand=batch_ligand, edge_ligand_idx=edge_ligand_idx, ligand_vec=ligand_vec,
                h_residue=h_residue, pos_residue=pos_residue, batch_residue=batch_residue, edge_residue_idx=edge_residue_idx,
                edge_diff=edge_diff, residue_frame=residue_frame)
            ligand_vec = ligand_vec + dligand_vec

        ligand_vec_output = self.last_layers(ligand_vec).squeeze(dim=2) + ligand_vec_input

        return ligand_vec_output
