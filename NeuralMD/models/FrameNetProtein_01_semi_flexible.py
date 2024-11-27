import torch
from torch import nn

from torch_geometric.nn import radius_graph
from torch_scatter import scatter

from .FrameNet_support import _normalize, MLP, weight_initialization, RBF_repredding_01, RBF_repredding_02


class FrameNetProtein01(nn.Module):
    def __init__(
        self,
        emb_dim, num_residue_acid,
        num_radial, cutoff, rbf_type, rbf_gamma,
        readout,
        batch_norm=True, momentum=0.2, dropout=0.2,
    ):
        super(FrameNetProtein01, self).__init__()
        self.emb_dim = emb_dim
        self.num_residue_acid = num_residue_acid
        self.cutoff = cutoff
        self.batch_norm = batch_norm
        self.momentum = momentum
        self.dropout = dropout
        self.readout = readout

        self.residue_edge_repredding = nn.Embedding(self.num_residue_acid, self.emb_dim)

        if rbf_type == "RBF_repredding_01":
            self.ligand_radial_embedding = nn.Sequential(
                RBF_repredding_01(num_radial, self.cutoff),
                nn.Linear(num_radial, emb_dim),
            )
            self.backbone_radial_embedding = nn.Sequential(
                RBF_repredding_01(num_radial, self.cutoff),
                nn.Linear(num_radial, emb_dim),
            )
        elif rbf_type == "RBF_repredding_02":
            self.ligand_radial_embedding = nn.Sequential(
                RBF_repredding_02(start=0, stop=self.cutoff, num_radial=num_radial, gamma=rbf_gamma),
                nn.Linear(num_radial, emb_dim),
            )
            self.backbone_radial_embedding = nn.Sequential(
                RBF_repredding_02(start=0, stop=self.cutoff, num_radial=num_radial, gamma=rbf_gamma),
                nn.Linear(num_radial, emb_dim),
            )

        # dim_list = [emb_dim, emb_dim, emb_dim, emb_dim, emb_dim]
        dim_list = [emb_dim, emb_dim]
        self.edge_layers_01 = MLP(
            dim_list=dim_list, batch_norm=False, dropout=dropout, momentum=momentum
            # dim_list=dim_list, batch_norm=batch_norm, dropout=dropout, momentum=momentum
        )

        # dim_list = [emb_dim, 2*emb_dim, emb_dim, emb_dim, emb_dim]
        dim_list = [emb_dim, 2*emb_dim, emb_dim]
        self.edge_layers = MLP(
            dim_list=dim_list, batch_norm=batch_norm, dropout=dropout, momentum=momentum
        )

        self.scalarization_layers = nn.Sequential(
            nn.Linear(3, 32),
            nn.SiLU(),
            nn.Linear(32, 1),
        )

        self.backbone_embedding = nn.Embedding(3, self.emb_dim)

        self.reset_parameters()
        return

    def reset_parameters(self):
        self.edge_layers_01.reset_parameters()
        self.edge_layers.reset_parameters()

        for i, layer in enumerate(self.ligand_radial_embedding):
            if isinstance(layer, nn.Linear):
                weight_initialization(layer)

        for i, layer in enumerate(self.scalarization_layers):
            if isinstance(layer, nn.Linear):
                weight_initialization(layer) 
        return

    def get_ligand_equivariant_representation(self, x, pos, batch):
        """
        The input includes 3 type of atoms, i.e., the backbone atoms. Thus, the num of backbone atom is three times the number of residue.
        
        pos: atom position [num_ligand, 3]
        x: atom type [num_ligand, emb_dim]
        """
        num_node = x.size()[0]
        edge_index = radius_graph(pos, r=self.cutoff, batch=batch)
        i, j = edge_index

        # [num_edge, 3]
        vec = pos[i] - pos[j]
        # [num_edge]
        dist = torch.norm(vec, dim=-1)
        # [num_edge, emb_dim]
        radial_repr = self.ligand_radial_embedding(dist)
        
        # [num_edge, emb_dim]
        x_j = x[j] * radial_repr
        # [num_edge, emb_dim]
        x_i_j = x[i] + x_j

        # [num_node, emb_dim]
        x = scatter(x_i_j, i, dim=0, dim_size=num_node, reduce=self.readout)
        return x

    def backbone_MPNN(self, vec, radial, backbone_atom_repr_u, backbone_atom_repr_v):
        """
        vec: [num_residue, 3]
        radial: [num_residue, emb_dim]
        backbone_atom_repr_u: [num_residue, emb_dim]
        backbone_atom_repr_v: [num_residue, latetn_dim]
        """
        # [num_residue, 1, emb_dim]
        backbone_atom_repr_u = backbone_atom_repr_u.unsqueeze(1)
        # [num_residue, 1, emb_dim]
        backbone_atom_repr_v = backbone_atom_repr_v.unsqueeze(1)

        # [num_residue, 3, emb_dim]
        edge_repr = vec.unsqueeze(2) * radial.unsqueeze(1) * backbone_atom_repr_u * backbone_atom_repr_v
        edge_repr = self.edge_layers_01(edge_repr)
        return edge_repr

    def get_backbone_edge_representation(self, dist, vec, backbone_atom_repr_u, backbone_atom_repr_v, backbone_frame):
        # num_residual, emb_dim
        radial_repr = self.backbone_radial_embedding(dist)
        # [num_residue, 3, emb_dim]
        residue_edge_repr = self.backbone_MPNN(vec, radial_repr, backbone_atom_repr_u, backbone_atom_repr_v)
        # [num_residue, 3, 1, emb_dim]
        residue_edge_repr = residue_edge_repr.unsqueeze(2)
        # [num_residue, 3, 3, emb_dim]
        scalarization = residue_edge_repr * backbone_frame.unsqueeze(3)
        # [num_residue, 3, emb_dim]
        scalarization = scalarization.sum(dim=1)

        # # Only requires this for E(3)-equivariant
        # scalarization[:, 2, :] = torch.abs(scalarization[:, 2, :].clone())

        # [num_residue, emb_dim, 3]
        scalarization = torch.permute(scalarization, (0, 2, 1))
        # [num_residue, emb_dim]
        scalarization = self.scalarization_layers(scalarization).squeeze(2) + scalarization[:, :, 0]
        # [num_residue, emb_dim]
        scalarization = self.edge_layers(scalarization)
        return scalarization

    def forward(self, pos_N, pos_Ca, pos_C, residue_type, batch):
        num_residue = residue_type.size()[0]

        # pos_center = (torch.sum(pos_N, dim=0, keepdim=True) + 
        #         torch.sum(pos_Ca, dim=0, keepdim=True) + 
        #         torch.sum(pos_C, dim=0, keepdim=True)) / (num_residue * 3)
        # pos_N = pos_N - pos_center
        # pos_Ca = pos_Ca - pos_center
        # pos_C = pos_C - pos_center

        # num_residue, emb_dim
        residue_type_repr = self.residue_edge_repredding(residue_type)

        backbone_idx = torch.LongTensor([0, 1, 2]).to(pos_N.device)
        # [3, emb_dim]
        backbone_type_repr = self.backbone_embedding(backbone_idx)
        # [1, 3, emb_dim]
        ligand_type_repr = backbone_type_repr.unsqueeze(0)
        # [num_residue, 3, emb_dim]
        ligand_type_repr = ligand_type_repr.expand([num_residue, -1, -1]).contiguous()
        # [num_residue * 3, emb_dim]
        ligand_type_repr = ligand_type_repr.view(-1, self.emb_dim)

        # [num_residue, 3, 3]
        pos = torch.stack([pos_N, pos_Ca, pos_C], dim=1)
        # [num_residue * 3, 3]
        pos = pos.view(-1, 3)

        # [num_residue * 3]
        expanded_batch = batch.unsqueeze(0).expand([3, -1]).contiguous().view(-1)

        # [num_residue * 3, self.latent]
        ligand_repr = self.get_ligand_equivariant_representation(x=ligand_type_repr, pos=pos, batch=expanded_batch)
        # [num_residue, 3, self.latent]
        ligand_repr = ligand_repr.view(num_residue, 3, self.emb_dim)
        # [3, num_residue, self.latent]
        ligand_repr = ligand_repr.permute([1, 0, 2])

        ##### get residue-level repr #####
        vec_N_Ca = pos_Ca - pos_N
        dist_N_Ca = torch.norm(vec_N_Ca, dim=-1)
        frame_N_Ca = _normalize(vec_N_Ca, dist_N_Ca)
        
        vec_Ca_C = pos_C - pos_Ca
        dist_Ca_C = torch.norm(vec_Ca_C, dim=-1)
        frame_Ca_C = _normalize(vec_Ca_C, dist_Ca_C)

        vec_cross = torch.cross(frame_N_Ca, frame_Ca_C)
        frame_cross = _normalize(vec_cross)

        # [num_residue, 3-direction or 3-basis, 3-element or 1-vec for each direction]
        backbone_frame = torch.stack([frame_N_Ca, frame_Ca_C, frame_cross], dim=2)

        # [num_residue, emb_dim]
        scalarization_N_Ca = self.get_backbone_edge_representation(
            dist=dist_N_Ca, vec=vec_N_Ca,
            backbone_atom_repr_u=ligand_repr[0], backbone_atom_repr_v=ligand_repr[1], backbone_frame=backbone_frame)
        # [num_residue, emb_dim]
        scalarization_Ca_C = self.get_backbone_edge_representation(
            dist=dist_Ca_C, vec=vec_Ca_C,
            backbone_atom_repr_u=ligand_repr[1], backbone_atom_repr_v=ligand_repr[2], backbone_frame=backbone_frame)

        # [num_residue, emb_dim]
        residue_repr = (scalarization_N_Ca + scalarization_Ca_C) / 2 + residue_type_repr

        return residue_repr