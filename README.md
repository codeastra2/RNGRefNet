# Road Network Refinemwnt using GNNs



## Pre Requisites

- Ensure to have run all steps in RNGDetplusplus repo.  
- Ensure to have downloaded the dataset and kept in `RNGDetPlusPlus\dataset\` folder. 
- Ensure to have the restuls of the run of RNGDetplusplus in:
    - `RNGDetPlusPlus/RNGDet_multi_ins/test`
- Install all required dependencies using 
    - `conda env create --name gnn --file environment.yml`

## Dataset Generation

### Custom Dataset generation 
- Run  `python gen_graph_data.py`.
- Pre generated dataset is present here: `/data/ushelf/mf-pba/data/ramd_rm/Best_Custom_Dset_32x32`

### Dataset from results of RNGdetplusplus
- Run `python gnn_correction.py` to construct and generate the dataset from the RNGDetplusplus results.
- Pre generated dataset is present here: `/data/ushelf/mf-pba/data/ramd_rm/GNN_Dset_frm_RNGDet2023-02-20\ 13\:35\:02.006282/`

## Training the GNN model
- The required dataset and parameters can be tuned in the code for now. 
- Pre trained model is present here: `/data/ushelf/mf-pba/data/ramd_rm/best_custom_dset_gnn_model.pt`
- Results are stored in the folder `Reports/`

## Correcting the road graphs 

- Run again `python gnn_correction.py` to correct  dataset graph from the RNGDetplusplus results.(Note that, you may choose to comment out the construction and storing of dataset since you have already generated it)
- Results along with metrics and visualizations will be stored in the folder 
 `Results_Correction/`
 
 ## Sample Results
 - Some sample results are present at `/data/ushelf/mf-pba/data/ramd_rm/sample_results`
 - In the visualizations there please note that the the nodes marked with **red** and considered invalid, we may observe the quality by seeing the regions where they are marked valid and invalid. 