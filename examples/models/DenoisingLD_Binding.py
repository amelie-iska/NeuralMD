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


class DenoisingLD(nn.Module):
    def __init__(self, args):
        super(DenoisingLD, self).__init__()

        self.model_3d_ligand = args.model_3d_ligand
        self.model_3d_protein = args.model_3d_protein
        self.emb_dim = args.emb_dim


        self.beta_0 = args.diffusion_beta_min
        self.beta_1 = args.diffusion_beta_max
        self.num_diffusion_timesteps = args.num_diffusion_timesteps

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

    def marginal_prob(self, x, t):
        log_mean_coeff = -0.25 * t ** 2 * (self.beta_1 - self.beta_0) - 0.5 * t * self.beta_0
        mean = torch.exp(log_mean_coeff[:, None]) * x
        std = torch.sqrt(1. - torch.exp(2. * log_mean_coeff))
        return mean, std

    def forward(self, input_data, condition, t):
        ligand_positions = input_data

        EPSILON = 1e-6
        
        # Perterb pos
        pos_noise = torch.randn_like(ligand_positions)

        # sample variances
        node2graph = condition[1]
        num_graphs = node2graph.max().item() + 1
        time_step = torch.randint(0, self.num_diffusion_timesteps, size=(num_graphs // 2 + 1,), device=ligand_positions.device)
        time_step = torch.cat([time_step, self.num_diffusion_timesteps - time_step - 1], dim=0)[:num_graphs]  # (num_graph, )

        time_step = time_step / self.num_diffusion_timesteps * (1 - EPSILON) + EPSILON  # normalize to [0, 1]
        # time_step = time_step.squeeze(-1)
        t_pos = time_step.index_select(0, node2graph)  # (num_nodes, )
        mean_pos, std_pos = self.marginal_prob(ligand_positions, t_pos)
        pos_perturbed = mean_pos + std_pos[:, None] * pos_noise

        delta_pos = self.get_score(input_data, condition, t)

        loss_pos = torch.sum((pos_noise - delta_pos) ** 2, -1)  # (num_node)
        loss_pos = loss_pos.mean()

        return loss_pos

    def get_score(self, input_data, condition, t):

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

        ligand_vec = self.complex_model(
            ligand_repr=ligand_repr,
            ligand_vec_input=ligand_vec,
            pos_ligand=ligand_positions,
            batch_ligand=condition[1],
            residue_repr=residue_repr,
            pos_residue=condition[1 + protein_index],
            batch_residue=condition[4 + protein_index].long(),
        )

        delta_pos = ligand_vec
        return delta_pos
