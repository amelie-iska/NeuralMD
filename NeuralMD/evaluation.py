import numpy as np
import torch


def get_distance_at_one_snapshot(positions):
    distance = positions.unsqueeze(2) - positions.transpose(0, 1).unsqueeze(0)  # [N, 3, N]
    distance = torch.norm(distance, dim=1)  # [N, N]
    return distance


def get_pair_distance_at_one_snapshot(positions_a, positions_b):
    distance = positions_a.unsqueeze(2) - positions_b.transpose(0, 1).unsqueeze(0)  # [N, 3, N]
    distance = torch.norm(distance, dim=1)  # [N, N]
    return distance


def get_matching_list(traj_target, traj_pred, batch):
    device = traj_target.device
    num_snapshots = traj_target.shape[0]

    num_complex = batch.max().item() + 1
    matching_list = []

    for idx_mol in range(num_complex):
        mask = (batch == idx_mol).to(device)
        
        traj_target_single_complex = traj_target[:, mask]
        traj_pred_single_complex = traj_pred[:, mask]

        for i in range(num_snapshots):
            distance_traj_target = get_distance_at_one_snapshot(traj_target_single_complex[i])
            distance_traj_pred = get_distance_at_one_snapshot(traj_pred_single_complex[i])

            matching = torch.sqrt(torch.mean((distance_traj_target - distance_traj_pred) ** 2))
            matching_list.append(matching)

    return matching_list


def get_stability_list(traj_target, traj_pred, batch, stability_threshold=0.5):
    """
    https://github.com/kyonofx/MDsim/blob/main/observable.ipynb

    traj_target: [traj_num, atom_num (batch), 3]
    traj_pred: [traj_num, atom_num (batch), 3]
    batch: node2graph
    """
    device = traj_target.device
    num_snapshots = traj_target.shape[0]

    num_complex = batch.max().item() + 1
    stability_list = []

    for idx_mol in range(num_complex):
        mask = (batch == idx_mol).to(device)
        
        traj_target_single_complex = traj_target[:, mask]
        traj_pred_single_complex = traj_pred[:, mask]

        for i in range(num_snapshots):
            distance_traj_target = get_distance_at_one_snapshot(traj_target_single_complex[i])
            distance_traj_pred = get_distance_at_one_snapshot(traj_pred_single_complex[i])

            distance_gap = torch.abs(distance_traj_target - distance_traj_pred)
            distance_gap = distance_gap <= stability_threshold
            
            stability = 100. * torch.sum(distance_gap) / distance_gap.numel()
            stability_list.append(stability.item())

    return stability_list


# https://en.wikipedia.org/wiki/Covalent_radius#:~:text=The%20covalent%20radius%2C%20rcov,)%20%2B%20r(B).
covalent_radii_dict = {
    1:  0.31,
    5:  0.84,
    6:  0.69,
    7:  0.71,
    8:  0.66,
    9:  0.57,
    12: 1.41,
    13: 1.21,
    14: 1.11,
    15: 1.07,
    16: 1.05,
    17: 1.02,
    21: 1.7,
    22: 1.6,
    23: 1.53,
    24: 1.39,
    25: 1.39,
    26: 1.32,
    27: 1.26,
    28: 1.24,
    29: 1.32,
    30: 1.22,
    31: 1.22,
    32: 1.2,
    33: 1.19,
    34: 1.2,
    35: 1.2,
    36: 1.16,
    39: 1.9,
    40: 1.75,
    41: 1.64,
    42: 1.54,
    43: 1.47,
    44: 1.46,
    45: 1.42,
    46: 1.39,
    47: 1.45,
    48: 1.44,
    49: 1.42,
    50: 1.39,
    51: 1.39,
    52: 1.38,
    53: 1.39,
    54: 1.4,
    57: 2.07,
    58: 2.04,
    59: 2.03,
    60: 2.01,
    62: 1.98,
    63: 1.98,
    64: 1.96,
    65: 1.94,
    66: 1.92,
    67: 1.92,
    68: 1.89,
    69: 1.9,
    70: 1.87,
    71: 1.87,
    72: 1.75,
    73: 1.7,
    74: 1.62,
    75: 1.51,
    76: 1.44,
    77: 1.41,
    78: 1.36,
    79: 1.36,
    80: 1.32,
    81: 1.45,
    82: 1.46,
    83: 1.48,
    90: 2.06,
    91: 2.00,
    92: 1.96,
    93: 1.9,
    94: 1.87,
}


def get_ligand_collision_list(traj_pred, atom_type, batch):
    device = traj_pred.device
    num_snapshots = traj_pred.shape[0]

    num_complex = batch.max().item() + 1
    collision_list = []

    covalent_radii_list = []
    for x in atom_type:
        covalent_radii_list.append(covalent_radii_dict[x.item()+1])
    covalent_radii_list = torch.FloatTensor(covalent_radii_list)
    
    for idx_mol in range(num_complex):
        mask = (batch == idx_mol).to(device)
        
        traj_pred_single_complex = traj_pred[:, mask]
        covalent_radii_single_complex = covalent_radii_list[mask]
        covalent_bond_distance = covalent_radii_single_complex.unsqueeze(0) + covalent_radii_single_complex.unsqueeze(1)
        covalent_bond_distance = covalent_bond_distance.to(device)

        for i in range(num_snapshots):
            distance_traj_pred = get_distance_at_one_snapshot(traj_pred_single_complex[i])

            distance_collision = covalent_bond_distance > distance_traj_pred

            collision = 100. * torch.sum(distance_collision) / distance_collision.numel()
            collision_list.append(collision.item())

    return collision_list


get_protein_collision_list = get_ligand_collision_list


def get_binding_collision_list_semi_flexible(traj_pred, ligand_atom_type, protein_pos, protein_atom_type, batch_ligand, batch_protein):
    device = traj_pred.device
    num_snapshots = traj_pred.shape[0]

    num_complex = batch_ligand.max().item() + 1
    collision_list = []

    ligand_covalent_radii_list = []
    for x in ligand_atom_type:
        ligand_covalent_radii_list.append(covalent_radii_dict[x.item()+1])
    ligand_covalent_radii_list = torch.FloatTensor(ligand_covalent_radii_list).to(device)

    protein_covalent_radii_list = []
    for x in protein_atom_type:
        protein_covalent_radii_list.append(covalent_radii_dict[x.item()+1])
    protein_covalent_radii_list = torch.FloatTensor(protein_covalent_radii_list).to(device)
    
    for idx_mol in range(num_complex):
        ligand_mask = (batch_ligand == idx_mol).to(device)
        protein_mask = (batch_protein == idx_mol).to(device)
        
        traj_pred_single_complex = traj_pred[:, ligand_mask]
        ligand_covalent_radii_single_complex = ligand_covalent_radii_list[ligand_mask]
        protein_pos_single_complex = protein_pos[protein_mask]
        protein_covalent_radii_single_complex = protein_covalent_radii_list[protein_mask]
        
        covalent_bond_distance = ligand_covalent_radii_single_complex.unsqueeze(1) + protein_covalent_radii_single_complex.unsqueeze(0)

        for i in range(num_snapshots):
            distance_pred = get_pair_distance_at_one_snapshot(traj_pred_single_complex[i], protein_pos_single_complex)
            distance_collision = covalent_bond_distance > distance_pred

            collision = 100. * torch.sum(distance_collision) / distance_collision.numel()
            collision_list.append(collision.item())
            
    return collision_list


def get_binding_collision_list_flexible(ligand_traj_pred, ligand_atom_type, protein_traj_pos, protein_atom_type, batch_ligand, batch_protein):
    device = ligand_traj_pred.device
    num_snapshots = ligand_traj_pred.shape[0]

    num_complex = batch_ligand.max().item() + 1
    collision_list = []

    ligand_covalent_radii_list = []
    for x in ligand_atom_type:
        ligand_covalent_radii_list.append(covalent_radii_dict[x.item()+1])
    ligand_covalent_radii_list = torch.FloatTensor(ligand_covalent_radii_list).to(device)

    protein_covalent_radii_list = []
    for x in protein_atom_type:
        protein_covalent_radii_list.append(covalent_radii_dict[x.item()+1])
    protein_covalent_radii_list = torch.FloatTensor(protein_covalent_radii_list).to(device)
    
    for idx_mol in range(num_complex):
        ligand_mask = (batch_ligand == idx_mol).to(device)
        protein_mask = (batch_protein == idx_mol).to(device)
        
        ligand_traj_pred_single_complex = ligand_traj_pred[:, ligand_mask]
        ligand_covalent_radii_single_complex = ligand_covalent_radii_list[ligand_mask]
        protein_pos_single_complex = protein_traj_pos[:, protein_mask]
        protein_covalent_radii_single_complex = protein_covalent_radii_list[protein_mask]
        
        covalent_bond_distance = ligand_covalent_radii_single_complex.unsqueeze(1) + protein_covalent_radii_single_complex.unsqueeze(0)

        for i in range(num_snapshots):
            distance_pred = get_pair_distance_at_one_snapshot(ligand_traj_pred_single_complex[i], protein_pos_single_complex[i])
            distance_collision = covalent_bond_distance > distance_pred

            collision = 100. * torch.sum(distance_collision) / distance_collision.numel()
            collision_list.append(collision.item())
            
    return collision_list