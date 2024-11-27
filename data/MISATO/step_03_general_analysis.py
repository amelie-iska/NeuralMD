import h5py
import os
import numpy as np
from tqdm import tqdm
import pickle
import importlib.resources
import NeuralMD.datasets.MISATO.utils

import matplotlib
import matplotlib.pyplot as plt
plt.switch_backend("agg")


with importlib.resources.files(NeuralMD.datasets.MISATO.utils) as resource_path:
    utils_dir = str(resource_path)
# utils_dir = os.path.join(os.path.abspath(os.path.dirname(__file__)), "dataset_MISATO_utils")
residueMap = pickle.load(open(os.path.join(utils_dir, "atoms_residue_map.pickle"),"rb"))
typeMap = pickle.load(open(os.path.join(utils_dir, "atoms_type_map.pickle"),"rb"))
nameMap = pickle.load(open(os.path.join(utils_dir, "atoms_name_map_for_pdb.pickle"),"rb"))

peptides_file = os.path.join(utils_dir, "peptides.txt")
peptides_idx_set = set()
with open(peptides_file) as f:
    for line in f.readlines():
        peptides_idx_set.add(line.strip().upper())


def update_residue_indices(i, type_string, atoms_type, atoms_residue, residue_name, residue_number, residue_atom_index, residue_Map, typeMap, molecules_begin_atom_index):
    if i < len(atoms_type)-1:
        if type_string == "O" and typeMap[atoms_type[i+1]] == "N" or residue_Map[atoms_residue[i+1]]=="MOL":
            # GLN has a O N sequence within the AA
            if not ((residue_name == "GLN" and residue_atom_index==12) or (residue_name == "ASN" and residue_atom_index==9)):
                residue_number +=1
                residue_atom_index = 0
    
    if i+1 in molecules_begin_atom_index:
        residue_number +=1
        residue_atom_index = 0

    return residue_number, residue_atom_index


def get_atom_name(atoms_type, atoms_residue, atoms_number, typeMap, residueMap, nameMap, atomic_numbers_Map, molecules_begin_atom_index):
    residue_number = 1
    residue_atom_index = 0
    standard_name = []
    cur_residue = []
    track_residue = 1
    
    for i in range(len(atoms_type)):
        residue_atom_index += 1
        type_string = typeMap[atoms_type[i]]
        residue_name = residueMap[atoms_residue[i]]
        try:
            atom_name = nameMap[(residue_name, residue_atom_index-1, type_string)]
        except KeyError:
            atom_name = atomic_numbers_Map[atoms_number[i]]+str(residue_atom_index)

        if track_residue == residue_number:
            cur_residue.append(atom_name)
        else:
            cur_residue = np.array(cur_residue)
            if np.count_nonzero(cur_residue == "CA") == np.count_nonzero(cur_residue == "C") and np.count_nonzero(cur_residue == "N") == np.count_nonzero(cur_residue == "C"):
                standard_name.append(cur_residue)
            else:
                standard_name.append(np.full(len(cur_residue), "X", dtype=str))
            cur_residue = [atom_name]
            track_residue = residue_number
        
        residue_number, residue_atom_index = update_residue_indices(i, type_string, atoms_type, atoms_residue, residue_name, residue_number, residue_atom_index, residueMap, typeMap, molecules_begin_atom_index)

    # handle the last residue
    cur_residue = np.array(cur_residue)
    if np.count_nonzero(cur_residue == "CA") == np.count_nonzero(cur_residue == "C") and np.count_nonzero(cur_residue == "N") == np.count_nonzero(cur_residue == "C"):
        standard_name.append(cur_residue)
    else:
        standard_name.append(np.full(len(cur_residue), "X", dtype=str))

    flattened_standard_name = [atom for residue in standard_name for atom in residue]
    return np.array(flattened_standard_name)


def extract_backbone(atoms_type, atoms_residue, atoms_number, typeMap, residueMap, nameMap, atomic_numbers_Map, molecules_begin_atom_index):
    atoms_type = get_atom_name(atoms_type, atoms_residue, atoms_number, typeMap, residueMap, nameMap, atomic_numbers_Map, molecules_begin_atom_index)
    mask_backbone = (atoms_type == "CA") | (atoms_type == "C") | (atoms_type == "N")

    protein_backbone_atom_type = atoms_type[mask_backbone]
    mask_ca = protein_backbone_atom_type == "CA"
    mask_c = protein_backbone_atom_type == "C"
    mask_n = protein_backbone_atom_type == "N"
    assert np.sum(mask_ca) == np.sum(mask_c) == np.sum(mask_n)
    return mask_backbone, mask_ca, mask_c, mask_n
    

def centroid(A):
    A = A.mean(axis=0)
    return A


def kabsch(coord_var, coord_ref):
    """
    calculation of Rotation Matrix R
    see SVD  http://en.wikipedia.org/wiki/Kabsch_algorithm
    and  proper/improper rotation, JCC 2004, 25, 1894.
    """
    covar = np.dot(coord_var.T, coord_ref)
    v, s, wt = np.linalg.svd(covar)
    d = (np.linalg.det(v) * np.linalg.det(wt)) < 0.0
    if d: # antialigns of the last singular vector
        s[-1] = -s[-1]
        v[:, -1] = -v[:, -1]
    R = np.dot(v, wt)
    return R


def align_frame_to_ref(trajectory_pos, varframe, coord_ref):
    """
    Gets coordinates, translates by centroid and rotates by rotation matrix R
    """
    coord_var = trajectory_pos[:, varframe]
    trans = centroid(coord_ref)
    coord_var_cen = coord_var - centroid(coord_var)
    coord_ref_cen = coord_ref - centroid(coord_ref)
    R = kabsch(coord_var_cen, coord_ref_cen)
    coord_var_shifted = np.dot(coord_var_cen,R) + trans
    return coord_var_shifted


def get_adaptability(trajectory_pos):
    ref = trajectory_pos[:, 0]
    NAtom = len(ref)
    dist_to_ref_mat = np.zeros((NAtom, 100))
    for ind in range(100):
        aligned = align_frame_to_ref(trajectory_pos, ind, ref)
        squared_dist = np.sum((ref-aligned)**2, axis=1)
        dist_to_ref_mat[:, ind] = np.sqrt(squared_dist)
    return dist_to_ref_mat


def get_trajectory_shift_distance(trajectory_pos):
    trajectory_shift_distance_list = []
    for i in range(99):
        pos_shift = trajectory_pos[:, i] - trajectory_pos[:, i+1]
        dist = np.linalg.norm(pos_shift, keepdims=False, axis=1)
        trajectory_shift_distance_list.append(dist)
    trajectory_shift_distance_list = np.concatenate(trajectory_shift_distance_list)
    return trajectory_shift_distance_list


def process(misato_data):
    atomic_numbers_Map = {1:"H", 5:"B", 6:"C", 7:"N", 8:"O", 9:"F", 11:"Na", 12:"Mg", 13:"Al", 14:"Si", 15:"P", 16:"S", 17:"Cl", 19:"K", 20:"Ca", 34:"Se", 35:"Br", 53:"I"}

    # TODO: need to double-check. Now the last molecules_begin_atom_index is treated as the starting point for small molecules.
    small_molecule_begin_index = misato_data["molecules_begin_atom_index"][:][-1]

    atoms_number = misato_data["atoms_number"][:]  # num_atom
    atoms_type = misato_data["atoms_type"][:]  # num_atom
    atoms_residue = misato_data["atoms_residue"][:]  # num_atom

    frames_interaction_energy = misato_data["frames_interaction_energy"][:] # 100

    trajectory_coordinates = misato_data["trajectory_coordinates"][:]  # 100, num_atom, 3
    trajectory_coordinates = np.transpose(trajectory_coordinates, (1, 0, 2)) # num_atom, 100, 3

    # [num_atom, 100, 3]
    protein_trajectory_coordinates = trajectory_coordinates[:small_molecule_begin_index]
    print("protein_trajectory_coordinates", protein_trajectory_coordinates.shape)

    mask_backbone, mask_ca, mask_c, mask_n = extract_backbone(atoms_type[:small_molecule_begin_index], atoms_residue[:small_molecule_begin_index], atoms_number[:small_molecule_begin_index], typeMap, residueMap, nameMap, atomic_numbers_Map, misato_data["molecules_begin_atom_index"][:][:-1])
    # [num_protein_backbone_atom, 100, 3]
    protein_backbone_coordinates = protein_trajectory_coordinates[mask_backbone, :]
    num_residue = protein_backbone_coordinates.shape[0]
    ca_adaptability = get_adaptability(protein_backbone_coordinates[mask_ca])
    c_adaptability = get_adaptability(protein_backbone_coordinates[mask_c])
    n_adaptability = get_adaptability(protein_backbone_coordinates[mask_n])

    ca_trajectory_shift_distance_list = get_trajectory_shift_distance(protein_backbone_coordinates[mask_ca])
    c_trajectory_shift_distance_list = get_trajectory_shift_distance(protein_backbone_coordinates[mask_c])
    n_trajectory_shift_distance_list = get_trajectory_shift_distance(protein_backbone_coordinates[mask_n])

    # Get atom type for small molecules
    # [num_small_molecule_atom]
    small_molecule_atoms_number = atoms_number[small_molecule_begin_index:]
    valid_small_molecule_atoms_mask = small_molecule_atoms_number != 1 # ignore atom H
    small_molecule_atoms_number = small_molecule_atoms_number[valid_small_molecule_atoms_mask]

    # [num_atom, 100, 3]
    small_molecule_trajectory_coordinates = trajectory_coordinates[small_molecule_begin_index:][valid_small_molecule_atoms_mask]
    small_molecule_adaptability = get_adaptability(small_molecule_trajectory_coordinates)
    small_molecule_trajectory_shift_distance = get_trajectory_shift_distance(small_molecule_trajectory_coordinates)

    num_atom = small_molecule_atoms_number.shape[0]

    protein_adaptability_list = [ca_adaptability, c_adaptability, n_adaptability]
    protein_adaptability_array = np.concatenate(protein_adaptability_list, axis=0)
    protein_trajectory_shift_distance = [ca_trajectory_shift_distance_list, c_trajectory_shift_distance_list, n_trajectory_shift_distance_list]
    protein_trajectory_shift_distance = np.concatenate(protein_trajectory_shift_distance, axis=0)
    ligand_adaptability_array = small_molecule_adaptability
    ligand_molecule_trajectory_shift_distance = small_molecule_trajectory_shift_distance
    # # [num_atom]
    # adaptability_array = np.mean(adaptability_array, axis=1)

    return num_residue, num_atom, frames_interaction_energy, protein_adaptability_array, ligand_adaptability_array, protein_trajectory_shift_distance, ligand_molecule_trajectory_shift_distance


def plot_distribution(value_list, file_name, label):

    plt.hist(value_list, color="blue", edgecolor="black", bins=20)

    plt.ylabel("Count", fontsize=20)
    plt.xlabel(label, fontsize=15)
    plt.savefig(file_name, dpi=500, bbox_inches="tight")
    plt.clf()
    return


if __name__ == "__main__":
    ##### load MD index
    file_name = "./raw/train_MD.txt"
    train_id_list = []
    f = open(file_name, "r")
    for line in f.readlines():
        line = line.strip()
        train_id_list.append(line)
    
    file_name = "./raw/val_MD.txt"
    val_id_list = []
    f = open(file_name, "r")
    for line in f.readlines():
        line = line.strip()
        val_id_list.append(line)
    
    file_name = "./raw/test_MD.txt"
    test_id_list = []
    f = open(file_name, "r")
    for line in f.readlines():
        line = line.strip()
        test_id_list.append(line)

    print(len(train_id_list), len(val_id_list), len(test_id_list))

    ##### load MD file
    file_name = "./raw/MD.hdf5"
    MD_file = h5py.File(file_name, "r")
    
    num_residue_list, num_atom_list, energy_list, protein_adaptability_list, ligand_adaptability_list, protein_trajectory_shift_distance_list, ligand_molecule_trajectory_shift_distance_list = [], [], [], [], [], [], []
    for c, train_id in enumerate(tqdm(train_id_list)):
        if train_id in peptides_idx_set:
            continue
        data = MD_file.get(train_id)
        num_residue, num_atom, energy, protein_adaptability, ligand_adaptability, protein_trajectory_shift_distance, ligand_molecule_trajectory_shift_distance = process(data)
        num_residue_list.append(num_residue)
        num_atom_list.append(num_atom)
        energy_list.append(energy)
        protein_adaptability_list.append(protein_adaptability)
        ligand_adaptability_list.append(ligand_adaptability)
        protein_trajectory_shift_distance_list.append(protein_trajectory_shift_distance)
        ligand_molecule_trajectory_shift_distance_list.append(ligand_molecule_trajectory_shift_distance)
        # if c >= 10:
        #     exit()
    num_residue_list = np.array(num_residue_list)
    num_atom_list = np.array(num_atom_list)
    energy_list = np.array(energy_list)
    protein_adaptability_list = np.concatenate(protein_adaptability_list)
    ligand_adaptability_list = np.concatenate(ligand_adaptability_list)
    protein_trajectory_shift_distance_list = np.concatenate(protein_trajectory_shift_distance_list)
    ligand_molecule_trajectory_shift_distance_list = np.concatenate(ligand_molecule_trajectory_shift_distance_list)
    np.savez("data_train",
        num_residue_list=num_residue_list, num_atom_list=num_atom_list, energy_list=energy_list,
        protein_adaptability_list=protein_adaptability_list, ligand_adaptability_list=ligand_adaptability_list,
        protein_trajectory_shift_distance_list=protein_trajectory_shift_distance_list, ligand_molecule_trajectory_shift_distance_list=ligand_molecule_trajectory_shift_distance_list)

    num_residue_list, num_atom_list, energy_list, protein_adaptability_list, ligand_adaptability_list, protein_trajectory_shift_distance_list, ligand_molecule_trajectory_shift_distance_list = [], [], [], [], [], [], []
    for c, val_id in enumerate(tqdm(val_id_list)):
        if val_id in peptides_idx_set:
            continue
        data = MD_file.get(val_id)
        num_residue, num_atom, energy, protein_adaptability, ligand_adaptability, protein_trajectory_shift_distance, ligand_molecule_trajectory_shift_distance = process(data)
        num_residue_list.append(num_residue)
        num_atom_list.append(num_atom)
        energy_list.append(energy)
        protein_adaptability_list.append(protein_adaptability)
        ligand_adaptability_list.append(ligand_adaptability)
        protein_trajectory_shift_distance_list.append(protein_trajectory_shift_distance)
        ligand_molecule_trajectory_shift_distance_list.append(ligand_molecule_trajectory_shift_distance)
        # if c >= 10:
        #     exit()
    num_residue_list = np.array(num_residue_list)
    num_atom_list = np.array(num_atom_list)
    energy_list = np.array(energy_list)
    protein_adaptability_list = np.concatenate(protein_adaptability_list)
    ligand_adaptability_list = np.concatenate(ligand_adaptability_list)
    protein_trajectory_shift_distance_list = np.concatenate(protein_trajectory_shift_distance_list)
    ligand_molecule_trajectory_shift_distance_list = np.concatenate(ligand_molecule_trajectory_shift_distance_list)
    np.savez("data_val",
        num_residue_list=num_residue_list, num_atom_list=num_atom_list, energy_list=energy_list,
        protein_adaptability_list=protein_adaptability_list, ligand_adaptability_list=ligand_adaptability_list,
        protein_trajectory_shift_distance_list=protein_trajectory_shift_distance_list, ligand_molecule_trajectory_shift_distance_list=ligand_molecule_trajectory_shift_distance_list)

    num_residue_list, num_atom_list, energy_list, protein_adaptability_list, ligand_adaptability_list, protein_trajectory_shift_distance_list, ligand_molecule_trajectory_shift_distance_list = [], [], [], [], [], [], []
    for c, test_id in enumerate(tqdm(test_id_list)):
        if test_id in peptides_idx_set:
            continue
        data = MD_file.get(test_id)
        num_residue, num_atom, energy, protein_adaptability, ligand_adaptability, protein_trajectory_shift_distance, ligand_molecule_trajectory_shift_distance = process(data)
        num_residue_list.append(num_residue)
        num_atom_list.append(num_atom)
        energy_list.append(energy)
        protein_adaptability_list.append(protein_adaptability)
        ligand_adaptability_list.append(ligand_adaptability)
        protein_trajectory_shift_distance_list.append(protein_trajectory_shift_distance)
        ligand_molecule_trajectory_shift_distance_list.append(ligand_molecule_trajectory_shift_distance)
        # if c >= 10:
        #     exit()
    num_residue_list = np.array(num_residue_list)
    num_atom_list = np.array(num_atom_list)
    energy_list = np.array(energy_list)
    protein_adaptability_list = np.concatenate(protein_adaptability_list)
    ligand_adaptability_list = np.concatenate(ligand_adaptability_list)
    protein_trajectory_shift_distance_list = np.concatenate(protein_trajectory_shift_distance_list)
    ligand_molecule_trajectory_shift_distance_list = np.concatenate(ligand_molecule_trajectory_shift_distance_list)
    np.savez("data_test",
        num_residue_list=num_residue_list, num_atom_list=num_atom_list, energy_list=energy_list,
        protein_adaptability_list=protein_adaptability_list, ligand_adaptability_list=ligand_adaptability_list,
        protein_trajectory_shift_distance_list=protein_trajectory_shift_distance_list, ligand_molecule_trajectory_shift_distance_list=ligand_molecule_trajectory_shift_distance_list)

    # os.makedirs("dataset_MISATO_analysis", exist_ok=True)
    # plot_distribution(train_num_atom_list, "dataset_MISATO_analysis/train_num_atom_distribution.png", "Number of Atoms")
    # plot_distribution(val_num_atom_list, "dataset_MISATO_analysis/val_num_atom_distribution.png", "Number of Atoms")
    # plot_distribution(test_num_atom_list, "dataset_MISATO_analysis/test_num_atom_distribution.png", "Number of Atoms")
    # plot_distribution(num_atom_list, "dataset_MISATO_analysis/num_atom_distribution.png", "Number of Atoms")

    # plot_distribution(train_num_residue_list, "dataset_MISATO_analysis/train_num_residue_distribution.png", "Number of Residues")
    # plot_distribution(val_residue_atom_list, "dataset_MISATO_analysis/val_num_residue_distribution.png", "Number of Residues")
    # plot_distribution(test_num_residue_list, "dataset_MISATO_analysis/test_num_residue_distribution.png", "Number of Residues")
    # plot_distribution(num_residue_list, "dataset_MISATO_analysis/num_residue_distribution.png", "Number of Residues")
