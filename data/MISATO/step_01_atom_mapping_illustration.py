import pickle

import importlib.resources
import NeuralMD.datasets.MISATO.utils


if __name__ == "__main__":
    with importlib.resources.path(NeuralMD.datasets.MISATO.utils, 'atoms_type_map.pickle') as data_path:
        with open(data_path, 'rb') as handle:
            atoms_type_map = pickle.load(handle)
    for k, v in atoms_type_map.items():
        print(k, v)

    # Look for the key associated with the value "CA"
    print("CA encoding:", [key for key, value in atoms_type_map.items() if value == "CA"])
    print("N encoding:", [key for key, value in atoms_type_map.items() if value == "N"])
    print("C encoding:", [key for key, value in atoms_type_map.items() if value == "C"])
    print()
    