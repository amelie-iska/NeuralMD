import torch
import torch.nn as nn

from NeuralMD.models import FrameNetLigand01, FrameNetProtein01, FrameNetComplex01


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
            # layers.append(nn.LeakyReLU(0.1))
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


class NeuralMD_Binding02(nn.Module):
    def __init__(self, args):
        super(NeuralMD_Binding02, self).__init__()

        self.model_3d_ligand = args.model_3d_ligand
        self.model_3d_protein = args.model_3d_protein
        self.emb_dim = args.emb_dim
        self.velocity_refined_value_coefficient = args.NeuralMD_velocity_refined_value_coefficient
        self.T = torch.nn.Parameter(torch.tensor(1.))
        self.Boltzmann_constant = 1.38e-23
        self.gamma = torch.nn.Parameter(torch.tensor([[10., 10., 10.]]))

        node_class = 119
        num_tasks = 1

        self.ligand_model = FrameNetLigand01(
            hidden_channels=args.emb_dim,
            cutoff=args.FrameNet_cutoff,
            num_layers=args.FrameNet_num_layers,
            num_radial=args.FrameNet_num_radial,
        )
        self.sigma_model = FrameNetLigand01(
            hidden_channels=args.emb_dim,
            cutoff=args.FrameNet_cutoff,
            num_layers=args.FrameNet_num_layers,
            num_radial=args.FrameNet_num_radial,
        )
        
        if args.use_MLP_velocity:
            self.velocity_model = FrameNetLigand01(
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

        self.complex_model = FrameNetComplex01(
            latent_dim=args.emb_dim,
            num_layer=args.FrameNet_complex_layer,
            cutoff=args.FrameNet_cutoff,
            num_radial=args.FrameNet_num_radial,
            rbf_type=args.FrameNet_rbf_type,
            rbf_gamma=args.FrameNet_gamma,
            readout=args.FrameNet_readout,
        )

        return

    def forward(self, t, input_data, condition):
        """
        NeuralMD with Langevin Dynamics:
        ma = F - \gamma m v + \sqrt{2m \gamma k_B T} R
        """
        protein_index = -1

        velocity, ligand_positions = input_data
        
        # [num_atom, 3]
        _, ligand_vec, ligand_repr = self.ligand_model(
            z=condition[0],
            pos=ligand_positions,
            batch=condition[1],
            return_repr=True,
        )
        _, sigma = self.sigma_model(
            z=condition[0],
            pos=ligand_positions,
            batch=condition[1],
        )
        ligand_mass = condition[2]
        protein_index = 3
        
        residue_repr = self.protein_model(
            pos_N=condition[0 + protein_index],
            pos_Ca=condition[1 + protein_index],
            pos_C=condition[2 + protein_index],
            residue_type=condition[3 + protein_index].long(),
            batch=condition[4 + protein_index].long(),
        )
        ligand_vec = self.complex_model(
            ligand_repr=ligand_repr,
            ligand_vec_input=ligand_vec,
            pos_ligand=ligand_positions,
            batch_ligand=condition[1],
            residue_repr=residue_repr,
            pos_residue=condition[1 + protein_index],
            batch_residue=condition[4 + protein_index].long(),
        )

        # [num_atom, 3]
        m = ligand_mass.unsqueeze(1)
        white_noise = torch.randn_like(m)
        # F = ligand_vec - self.gamma * m * velocity_refined_value + torch.sqrt(2 * m * self.gamma * self.Boltzmann_constant * self.T) * white_noise
        F = ligand_vec + self.gamma * white_noise * sigma
        acceleration = F / m

        output = (acceleration, velocity)

        return output
