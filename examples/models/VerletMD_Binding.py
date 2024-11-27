import torch
import torch.nn as nn

from NeuralMD.models import FrameNetLigand01, FrameNetProtein01, FrameNetComplexEnergy01
from torch.autograd import grad
from torch_scatter import scatter


def weight_initialization(layer):
    # torch.nn.init.xavier_uniform_(layer.weight)
    torch.nn.init.kaiming_uniform_(layer.weight)
    return


class MLP(nn.Module):
    def __init__(self, dim_list, batch_norm, dropout, momentum):
        super(MLP, self).__init__()

        layers = []
        for input_dim, output_dim in zip(dim_list[:-1], dim_list[1:]):
            if batch_norm:
                layers.append(nn.BatchNorm1d(input_dim, momentum=momentum))
            layers.append(nn.SiLU())
            layers.append(nn.Dropout(dropout))
            layers.append(nn.Linear(input_dim, output_dim))

        self.layers = nn.Sequential(*layers)

        self.reset_parameters()
        return

    def reset_parameters(self):
        for i, layer in enumerate(self.layers):
            if isinstance(layer, nn.Linear):
                weight_initialization(layer) 
        return
    
    def forward(self, x):
        return self.layers(x)


class VerletMD(nn.Module):
    def __init__(self, args):
        super(VerletMD, self).__init__()

        self.model_3d_ligand = args.model_3d_ligand
        self.model_3d_protein = args.model_3d_protein
        self.emb_dim = args.emb_dim

        node_class = 119
        num_tasks = 1

        self.ligand_model = FrameNetLigand01(
            hidden_channels=args.emb_dim,
            cutoff=args.FrameNet_cutoff,
            num_layers=args.FrameNet_num_layers,
            num_radial=args.FrameNet_num_radial,
        )
        
        self.protein_model = FrameNetProtein01(
            emb_dim=args.emb_dim, num_residue_acid=26,
            num_radial=args.FrameNet_num_radial, cutoff=args.FrameNet_cutoff,
            rbf_type=args.FrameNet_rbf_type, rbf_gamma=args.FrameNet_gamma,
            readout=args.FrameNet_readout,
        )

        self.complex_model = FrameNetComplexEnergy01(
            latent_dim=args.emb_dim,
            num_layer=args.FrameNet_complex_layer,
            cutoff=args.FrameNet_cutoff,
            num_radial=args.FrameNet_num_radial,
            rbf_type=args.FrameNet_rbf_type,
            rbf_gamma=args.FrameNet_gamma,
            readout=args.FrameNet_readout,
        )

        self.last_layer = nn.Linear(args.emb_dim, 1)

        return

    def forward(self, input_data, condition):

        ligand_positions = input_data

        # [num_atom, 3], [num_atom, latent_dim]
        _, ligand_vec, ligand_repr = self.ligand_model(
            z=condition[0],
            pos=ligand_positions,
            batch=condition[1],
            return_repr=True,
        )
        
        protein_index = 3
        residue_repr = self.protein_model(
            pos_N=condition[0 + protein_index],
            pos_Ca=condition[1 + protein_index],
            pos_C=condition[2 + protein_index],
            residue_type=condition[3 + protein_index].long(),
            batch=condition[4 + protein_index].long(),
        )

        ligand_repr = self.complex_model(
            ligand_repr=ligand_repr,
            ligand_vec_input=ligand_vec,
            pos_ligand=ligand_positions,
            batch_ligand=condition[1],
            residue_repr=residue_repr,
            pos_residue=condition[1 + protein_index],
            batch_residue=condition[4 + protein_index].long(),
        )

        complex_repr = scatter(ligand_repr, condition[1], dim=0)

        energy = self.last_layer(complex_repr.mean(dim=0))

        return energy

    def move(self, input_data, condition, delta_t=1):
        ligand_positions, ligand_velocities = input_data
        ligand_positions.requires_grad_()

        energy = self.forward(ligand_positions, condition)

        force = -grad(outputs=energy, inputs=ligand_positions, grad_outputs=torch.ones_like(energy), create_graph=True, retain_graph=True)[0]

        mass = condition[2]
        a = force / mass.unsqueeze(1)

        ligand_velocities = ligand_velocities + 0.5 * a * delta_t
        ligand_positions = ligand_positions + ligand_velocities * delta_t + 0.5 * a * delta_t**2

        return ligand_positions, ligand_velocities
