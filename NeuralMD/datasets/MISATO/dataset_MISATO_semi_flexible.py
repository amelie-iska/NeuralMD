import os
import numpy as np
import h5py
from tqdm import tqdm
import pickle

import torch
from torch.utils.data import Dataset
from torch_geometric.data import Data, InMemoryDataset
from .common import extract_backbone

utils_dir = os.path.join(os.path.abspath(os.path.dirname(__file__)), "utils")


def parse_MISATO_data(misato_data, atom_index2name_dict, atom_num2atom_mass, residue_index2name_dict, protein_atom_index2standard_name_dict, atom_reisdue2standard_atom_name_dict):
    ligand_begin_index = misato_data["molecules_begin_atom_index"][:][-1]

    atom_index = misato_data["atoms_number"][:]
    protein_atom_index = misato_data["atoms_type"][:]
    residue_index = misato_data["atoms_residue"][:]

    frames_interaction_energy = np.expand_dims(misato_data["frames_interaction_energy"][:], 0) # 1, 100

    ##### get coordinates (remove center) #####
    trajectory_coordinates = misato_data["trajectory_coordinates"][:]  # 100, num_atom, 3
    trajectory_coordinates = np.transpose(trajectory_coordinates, (1, 0, 2)) # num_atom, 100, 3
    
    trajectory_coordinates_flattened = np.copy(trajectory_coordinates)
    trajectory_coordinates_flattened = trajectory_coordinates_flattened.reshape(-1, 3) # num_atom * 100, 3
    trajectory_pos_center = np.sum(trajectory_coordinates_flattened, axis=0) / trajectory_coordinates_flattened.shape[0]

    trajectory_coordinates = trajectory_coordinates - trajectory_pos_center
    protein_coordinates = trajectory_coordinates[:ligand_begin_index, 0]
    ##################################################

    mask_backbone, mask_ca, mask_c, mask_n = extract_backbone(protein_atom_index[:ligand_begin_index], residue_index[:ligand_begin_index], atom_index[:ligand_begin_index], protein_atom_index2standard_name_dict, residue_index2name_dict, atom_reisdue2standard_atom_name_dict, atom_index2name_dict, misato_data["molecules_begin_atom_index"][:][-1:])
    # [num_protein_backbone_atom, 3]
    protein_backbone_coordinates = protein_coordinates[mask_backbone, :]
    protein_backbone_coordinates = torch.tensor(protein_backbone_coordinates, dtype=torch.float32)
    # [num_protein_backbone_atom]
    protein_backbone_residue = residue_index[:ligand_begin_index][mask_backbone][mask_ca]
    protein_backbone_residue = torch.tensor(protein_backbone_residue, dtype=torch.int64)
    assert protein_backbone_residue.min() >= 1
    protein_backbone_residue -= 1
    # [num_protein_backbone_atom]
    mask_ca = torch.tensor(mask_ca, dtype=torch.bool)
    mask_c = torch.tensor(mask_c, dtype=torch.bool)
    mask_n = torch.tensor(mask_n, dtype=torch.bool)

    # Get atom type for small molecules
    # [num_ligand_atom]
    ligand_atom_index = atom_index[ligand_begin_index:]
    valid_ligand_atoms_mask = ligand_atom_index != 1 # ignore atom H
    ligand_atom_index = ligand_atom_index[valid_ligand_atoms_mask]
    ligand_atom_index -= 1 # index starting with 0

    # Get atom mass for small molecules
    ligand_atoms_mass = [atom_num2atom_mass[atom_num + 1] for atom_num in ligand_atom_index]
    
    # Get coordinates along the trajectory for small molecules
    # [num_ligand_atom, 100, 3]
    ligand_trajectory_coordinates = trajectory_coordinates[ligand_begin_index:][valid_ligand_atoms_mask]

    ligand_atom_index = torch.tensor(ligand_atom_index, dtype=torch.int64)
    ligand_atoms_mass = torch.tensor(ligand_atoms_mass, dtype=torch.float32)
    ligand_trajectory_coordinates = torch.tensor(ligand_trajectory_coordinates, dtype=torch.float32)

    # [1, 100]
    frames_interaction_energy = torch.tensor(frames_interaction_energy, dtype=torch.float32)

    data = Data(
        protein_pos=protein_backbone_coordinates,
        protein_backbone_residue=protein_backbone_residue,
        mask_ca=mask_ca,
        mask_c=mask_c,
        mask_n=mask_n,
        ligand_x=ligand_atom_index,
        ligand_mass=ligand_atoms_mass,
        ligand_trajectory_pos=ligand_trajectory_coordinates,
        energy=frames_interaction_energy,
    )
    return data


class DatasetMISATOSemiFlexibleMultiTrajectory(InMemoryDataset):
    def __init__(self, root, mode):
        self.root = root
        self.mode = mode
        self.c_alpha_atom_type = 6
        self.c_atom_type = 3
        self.n_atom_type = 24
        super(DatasetMISATOSemiFlexibleMultiTrajectory, self).__init__(root, None, None, None)

        self.data, self.slices = torch.load(self.processed_paths[0])
        return

    @property
    def raw_file_names(self):
        file_name = "MD.hdf5"
        return [file_name, "{}_MD.txt".format(self.mode)]

    @property
    def raw_dir(self):
        return os.path.join(self.root, "raw")

    @property
    def processed_file_names(self):
        return "geometric_data_processed_{}.pt".format(self.mode)

    @property
    def processed_dir(self):
        return os.path.join(self.root, "processed_semi_flexible")

    def process(self):
        split_file_path = self.raw_paths[1]
        f_ = open(split_file_path, "r")
        idx_list = []
        for line in f_.readlines():
            line = line.strip()
            idx_list.append(line)

        MD_file_path = self.raw_paths[0]
        MD_data = h5py.File(MD_file_path, "r")

        residue_index2name_dict = pickle.load(open(os.path.join(utils_dir, 'atoms_residue_map.pickle'),'rb'))
        protein_atom_index2standard_name_dict = pickle.load(open(os.path.join(utils_dir, 'atoms_type_map.pickle'),'rb'))
        atom_reisdue2standard_atom_name_dict = pickle.load(open(os.path.join(utils_dir, 'atoms_name_map_for_pdb.pickle'),'rb'))

        peptides_file = os.path.join(utils_dir, "peptides.txt")
        peptides_idx_set = set()
        with open(peptides_file) as f:
            for line in f.readlines():
                peptides_idx_set.add(line.strip().upper())
        
        atom_index2name_dict = {1:'H', 5:'B', 6:'C', 7:'N', 8:'O', 9:'F', 11:'Na', 12:'Mg', 13:'Al', 14:'Si', 15:'P', 16:'S', 17:'Cl', 19:'K', 20:'Ca', 34:'Se', 35:'Br', 53:'I'}

        import importlib.resources
        import NeuralMD.datasets
        import pandas as pd
        with importlib.resources.path(NeuralMD.datasets, 'periodic_table.csv') as file_name:
            periodic_table_file = file_name
        periodic_table_data = pd.read_csv(periodic_table_file)
        atom_num2atom_mass = {}
        for i in range(1, 119):
            atom_mass = periodic_table_data.loc[i-1]['AtomicMass']
            atom_num2atom_mass[i] = atom_mass

        data_list = []
        for idx in tqdm(idx_list):
            if idx in peptides_idx_set:
                continue

            misato_data = MD_data.get(idx)
            data = parse_MISATO_data(
                misato_data, atom_index2name_dict=atom_index2name_dict, atom_num2atom_mass=atom_num2atom_mass,
                residue_index2name_dict=residue_index2name_dict, protein_atom_index2standard_name_dict=protein_atom_index2standard_name_dict, atom_reisdue2standard_atom_name_dict=atom_reisdue2standard_atom_name_dict)
            data_list.append(data)
 
        data, slices = self.collate(data_list)
        print('Saving...')
        torch.save((data, slices), self.processed_paths[0])

        return


class DatasetMISATOSemiFlexibleSingleTrajectory(Dataset):
    def __init__(self, root, PDB_ID):
        self.root = root
        self.MD_file_path = os.path.join(root, "raw", "MD.hdf5")
        self.PDB_ID = PDB_ID
        self.process()

        return

    def process(self):
        MD_data = h5py.File(self.MD_file_path, "r")

        residue_index2name_dict = pickle.load(open(os.path.join(utils_dir, 'atoms_residue_map.pickle'),'rb'))
        protein_atom_index2standard_name_dict = pickle.load(open(os.path.join(utils_dir, 'atoms_type_map.pickle'),'rb'))
        atom_reisdue2standard_atom_name_dict = pickle.load(open(os.path.join(utils_dir, 'atoms_name_map_for_pdb.pickle'),'rb'))

        peptides_file = os.path.join(utils_dir, "peptides.txt")
        peptides_idx_set = set()
        with open(peptides_file) as f:
            for line in f.readlines():
                peptides_idx_set.add(line.strip().upper())

        atom_index2name_dict = {1:'H', 5:'B', 6:'C', 7:'N', 8:'O', 9:'F', 11:'Na', 12:'Mg', 13:'Al', 14:'Si', 15:'P', 16:'S', 17:'Cl', 19:'K', 20:'Ca', 34:'Se', 35:'Br', 53:'I'}

        import importlib.resources
        import NeuralMD.datasets
        import pandas as pd
        with importlib.resources.path(NeuralMD.datasets, 'periodic_table.csv') as file_name:
            periodic_table_file = file_name
        periodic_table_data = pd.read_csv(periodic_table_file)
        atom_num2atom_mass = {}
        for i in range(1, 119):
            atom_mass = periodic_table_data.loc[i-1]['AtomicMass']
            atom_num2atom_mass[i] = atom_mass

        self.data_list = []
        misato_data = MD_data.get(self.PDB_ID)
        data = parse_MISATO_data(
            misato_data, atom_index2name_dict=atom_index2name_dict, atom_num2atom_mass=atom_num2atom_mass,
            residue_index2name_dict=residue_index2name_dict, protein_atom_index2standard_name_dict=protein_atom_index2standard_name_dict, atom_reisdue2standard_atom_name_dict=atom_reisdue2standard_atom_name_dict)
        self.data_list.append(data)
        return

    def __len__(self):
        return 1

    def __getitem__(self, idx):
        data = self.data_list[idx]
        return data

if __name__ == "__main__":    
    data_root_list = ["../../data/MISATO_100"]
    mode_list = ["train", "val", "test"]
    for data_root in data_root_list:
        for mode in mode_list:
            dataset = DatasetMISATO(data_root, mode)
