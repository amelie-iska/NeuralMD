import random
import h5py
import os
from tqdm import tqdm
import importlib.resources
import NeuralMD.datasets.MISATO.utils


def write_splitting_idx_to_file(idx_list, file_name):
    f = open(file_name, "w")
    for idx in idx_list:
        print(idx, file=f)
    return


def generate_raw_data():
    for subsample_size in subsample_size_list:
        neo_folder = "../../data/MISATO_{}/raw".format(subsample_size)
        os.makedirs(neo_folder, exist_ok=True)
        neo_file_path = os.path.join(neo_folder, "MD.hdf5")

        f_out = h5py.File(neo_file_path, "w")

        target_key_list = []
        for key in key_list:
            if key in peptides_idx_set:
                continue
            target_key_list.append(key)
            if len(target_key_list) >= subsample_size:
                break
            

        for key in tqdm(target_key_list):
            data_group = f_in.get(key)
            f_in.copy(key, f_out)

        f_out.flush()
        f_out.close()

        train_idx = int(0.8 * subsample_size)
        val_idx = int(0.9 * subsample_size)
        train_file = os.path.join(neo_folder, "train_MD.txt")
        val_file = os.path.join(neo_folder, "val_MD.txt")
        test_file = os.path.join(neo_folder, "test_MD.txt")
        write_splitting_idx_to_file(target_key_list[:train_idx], train_file)
        write_splitting_idx_to_file(target_key_list[train_idx:val_idx], val_file)
        write_splitting_idx_to_file(target_key_list[val_idx:], test_file)
    return


def check_raw_subset_data():
    for subsample_size in subsample_size_list:
        neo_folder = "../MISATO_{}/raw".format(subsample_size)
        neo_file_path = os.path.join(neo_folder, "MD.hdf5")

        f_in = h5py.File(neo_file_path, "r")
        key_list = f_in.keys()
        key_list = [x for x in key_list]
        print("key_list", key_list)

        key = key_list[0]

        data = f_in[key]
        print(data.keys())

        atoms_element = data["atoms_element"]
        print(atoms_element, atoms_element[:])
        
        atoms_number = data["atoms_number"]
        print(atoms_number, atoms_number[:])
        
        atoms_residue = data["atoms_residue"]
        print(atoms_residue, atoms_residue[:])
        
        atoms_type = data["atoms_type"]
        print(atoms_type, atoms_type[:])

        frames_interaction_energy = data["frames_interaction_energy"]
        print(frames_interaction_energy, frames_interaction_energy[:])
        
        trajectory_coordinates = data["trajectory_coordinates"]
        print(trajectory_coordinates)
        print("\n\n\n")

    return


if __name__ == "__main__":
    random.seed(42)
    
    peptides_idx_set = set()
    with importlib.resources.path(NeuralMD.datasets.MISATO.utils, 'peptides.txt') as peptides_file, open(peptides_file, 'r') as f:
        for line in f.readlines():
            peptides_idx_set.add(line.strip().upper())
            
    ########## raw dataset subsaple #####
    file_name = "./raw/MD.hdf5"

    f_in = h5py.File(file_name, "r")
    key_list = f_in.keys()
    key_list = [x for x in key_list]
    random.shuffle(key_list)

    subsample_size_list = [100, 1000]

    generate_raw_data()
    f_in.close()

    check_raw_subset_data()
