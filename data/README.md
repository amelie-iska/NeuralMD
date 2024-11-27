## 1. MISATO

We take the following steps to prepare the MISATO dataset, `midir -p MISATO; cd MISATO`

First, please follow the instruction on the [MISATO GitHub repo](https://github.com/t7morgen/misato-dataset), and download data with `wget -O data/MD/h5_files/MD.hdf5 https://zenodo.org/record/7711953/files/MD.hdf5`. Then we can do `python step_02_subsample.py` to sample two subdatasets. The other three scripts are for analysis.
