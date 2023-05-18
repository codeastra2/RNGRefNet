# Road Network Refinement using GNNs

Using Graph Neural Networks to refine generated road networks. GNNs perform node 
classification, to remove invalid verties such as vertices on buildings, water bodies
and greeneries. 

Imitation learning explores the most generates the most roads however, it also
generates the most invalid vertices especially on water bodies. RNGDet is used a the example here. 

## Ground truth

Magenta represents valid points , red points represents invalid points. Cyan represents road network. 

![Invalid points on water](static/11_invver_gt.png)
![Invalid points on greenery](static/0_invver_pred.png)


## Prediction

Magenta represents predicted  valid points , red points represents predicted invalid points. Cyan represents road network. 

![Invalid points on water](static/11_invver_pred.png)
![Invalid points on greenery](static/0_invver_pred.png)


The following requires th road networks generated from RNGDet, and then the enerated dataset can be used for training this GNN, please replace the paths in respective filepaths. Some of hte pre trained models and datasets  are still behind license since this as done as a part of my master thesis at DLR (German Aerospace Center) Oberpfaffenhofen. 

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

### Dataset from results of RNGdetplusplus
- Run `python gnn_correction.py` to construct and generate the dataset from the RNGDetplusplus results.

## Training the GNN model
- The required dataset and parameters can be tuned in the code for now. 
- Results are stored in the folder `Reports/`

## Correcting the road graphs 

- Run again `python gnn_correction.py` to correct  dataset graph from the RNGDetplusplus results.(Note that, you may choose to comment out the construction and storing of dataset since you have already generated it)
- Results along with metrics and visualizations will be stored in the folder 
 `Results_Correction/`