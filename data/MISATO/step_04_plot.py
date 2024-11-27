import numpy as np

import seaborn as sns
import matplotlib
import matplotlib.pyplot as plt
plt.switch_backend("agg")
sns.set_theme(style="darkgrid")


if __name__ == "__main__":
    template = "data_{}.npz"
    mode_list = ["train", "val", "test"]
    for mode in mode_list:
        file_name = template.format(mode)
        data = np.load(file_name)
        # [num_mol]
        num_residue_list = data["num_residue_list"]
        # [num_mol]
        num_atom_list = data["num_atom_list"]
        # [num_mol, 100]
        energy_list = data["energy_list"]
        # [num_total_atom]
        protein_adaptability_list = data["protein_adaptability_list"]
        ligand_adaptability_list = data["ligand_adaptability_list"]
        protein_trajectory_shift_distance_list = data["protein_trajectory_shift_distance_list"]
        ligand_molecule_trajectory_shift_distance_list = data["ligand_molecule_trajectory_shift_distance_list"]
        print("energy_list", energy_list.shape)

        print("num_residue_list", num_residue_list)
        sns.histplot(data=num_residue_list, bins=50, binwidth=10)
        plt.savefig("figures/{}_num_residue.png".format(mode), dpi=500, bbox_inches="tight")
        plt.clf()
        
        print("num_atom_list", num_atom_list)
        sns.histplot(data=num_atom_list, bins=50, binwidth=1)
        plt.savefig("figures/{}_num_atom.png".format(mode), dpi=500, bbox_inches="tight")
        plt.clf()

        delta_energy_list = []
        for i in range(0, 100):
            delta_energy_list.append(energy_list[:, i] - energy_list[:, 0])
        delta_energy_list = np.array(delta_energy_list)
        print("delta_energy_list", delta_energy_list.shape)
        mean_delta_energy_list = np.mean(delta_energy_list, axis=1)
        std_delta_energy_list = np.std(delta_energy_list, axis=1)
        x_axis = np.arange(0, 100)
        ax = sns.lineplot(x=x_axis, y=mean_delta_energy_list, errorbar="sd")
        ax.fill_between(x_axis, mean_delta_energy_list+std_delta_energy_list, mean_delta_energy_list-std_delta_energy_list, alpha=0.2)
        plt.savefig("figures/{}_delta_energy.png".format(mode), dpi=500, bbox_inches="tight")
        plt.clf()

        print("protein_adaptability_list", protein_adaptability_list.shape)
        mean_protein_adaptability_list = np.mean(protein_adaptability_list, axis=1)
        print("mean_protein_adaptability_list", mean_protein_adaptability_list.shape)
        sns.histplot(data=mean_protein_adaptability_list, bins=50, binwidth=1)
        plt.savefig("figures/{}_protein_adaptability_atom.png".format(mode), dpi=500, bbox_inches="tight")
        plt.clf()

        mean_protein_adaptability_list = np.mean(protein_adaptability_list, axis=0)
        std_protein_adaptability_list = np.std(protein_adaptability_list, axis=0)
        x_axis = np.arange(0, 100)
        ax = sns.lineplot(x=x_axis, y=mean_protein_adaptability_list, errorbar="sd")
        ax.fill_between(x_axis, mean_protein_adaptability_list+std_protein_adaptability_list, mean_protein_adaptability_list-std_protein_adaptability_list, alpha=0.2)
        plt.savefig("figures/{}_protein_adaptability_snapshot.png".format(mode), dpi=500, bbox_inches="tight")
        plt.clf()

        print("ligand_adaptability_list", ligand_adaptability_list.shape)
        mean_ligand_adaptability_list = np.mean(ligand_adaptability_list, axis=1)
        print("mean_ligand_adaptability_list", mean_ligand_adaptability_list.shape)
        sns.histplot(data=mean_ligand_adaptability_list, bins=50, binwidth=1)
        plt.savefig("figures/{}_ligand_adaptability_atom.png".format(mode), dpi=500, bbox_inches="tight")
        plt.clf()

        mean_ligand_adaptability_list = np.mean(ligand_adaptability_list, axis=0)
        std_ligand_adaptability_list = np.std(ligand_adaptability_list, axis=0)
        x_axis = np.arange(0, 100)
        ax = sns.lineplot(x=x_axis, y=mean_ligand_adaptability_list, errorbar="sd")
        ax.fill_between(x_axis, mean_ligand_adaptability_list+std_ligand_adaptability_list, mean_ligand_adaptability_list-std_ligand_adaptability_list, alpha=0.2)
        plt.savefig("figures/{}_ligand_adaptability_snapshot.png".format(mode), dpi=500, bbox_inches="tight")
        plt.clf()

        limit, num_bins = 5, 50
        print("protein_trajectory_shift_distance_list", protein_trajectory_shift_distance_list.shape)
        protein_trajectory_shift_distance_list = [x for x in protein_trajectory_shift_distance_list if x <= limit]
        np.random.shuffle(protein_trajectory_shift_distance_list)
        if mode == "train":
            protein_trajectory_shift_distance_list = protein_trajectory_shift_distance_list[:len(protein_trajectory_shift_distance_list) // 10000]
        print("protein_trajectory_shift_distance_list", len(protein_trajectory_shift_distance_list))
        sns.histplot(data=protein_trajectory_shift_distance_list, bins=num_bins, binwidth=limit/num_bins)
        plt.savefig("figures/{}_protein_trajectory_shift_distance.png".format(mode), dpi=500, bbox_inches="tight")
        plt.clf()

        limit, num_bins = 5, 50
        print("ligand_molecule_trajectory_shift_distance_list", ligand_molecule_trajectory_shift_distance_list.shape)
        ligand_molecule_trajectory_shift_distance_list = [x for x in ligand_molecule_trajectory_shift_distance_list if x <= limit]
        np.random.shuffle(ligand_molecule_trajectory_shift_distance_list)
        if mode == "train":
            ligand_molecule_trajectory_shift_distance_list = ligand_molecule_trajectory_shift_distance_list[:len(ligand_molecule_trajectory_shift_distance_list) // 10000]
        print("ligand_molecule_trajectory_shift_distance_list", len(ligand_molecule_trajectory_shift_distance_list))
        sns.histplot(data=ligand_molecule_trajectory_shift_distance_list, bins=num_bins, binwidth=limit/num_bins)
        plt.savefig("figures/{}_ligand_trajectory_shift_distance.png".format(mode), dpi=500, bbox_inches="tight")
        plt.clf()
