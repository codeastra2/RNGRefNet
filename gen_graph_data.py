import torch
from torch_geometric.data import Data
import json
import numpy as np
from PIL import Image
import os
from pathlib import Path 
import random
import pickle 

# Set for reproducability
random.seed(42)

DATASET_STORE_PATH = Path("GNN_Dset_3vertex32x32")

# Data Structure to hold images and graph structure. 
IMG_DICT = {
    "sat_img": None,
    "p_sat_img": None,
}

GR_DICT = {
    "edge_list": [],
    "vertex_list": []
}

'''
So this will generate node features of 4*4 pixels where the GSD
is equal to 1m for every pixel so this is 16sqm and that should be enough area 
to classify whether a node belongs to a road or not. 
'''
pad_size = 48
# Tiling size of the areal image.  
tile_size = 2048
# Fraction of the number of valid samples to filter for further generating 
# the invalid vertices.
inv_smp_rat = 0.01
# Pixel distance between the invalid vertices.  
pix_dist = 40

def load_imgs_and_ev(img_name):
    """
    Function to load in the images and corresponding graph structure.
    """
    graph_path = "../dataset/graph/"

    IMG_DICT["sat_img"] = np.array(Image.open(os.path.join('../dataset',f'20cities/region_{img_name}_sat.png')))
    IMG_DICT["p_sat_img"] = np.pad(IMG_DICT["sat_img"],np.array(((pad_size,pad_size),(pad_size,pad_size),(0,0))),'constant')

    with open(f"{graph_path}/{img_name}.json",'r') as jf:
        jf_json = json.load(jf)
        GR_DICT["vertices_list"] = jf_json['vertices']
        GR_DICT["edge_list"] = jf_json['edges']
        GR_DICT["vcoord_list"] = []
        GR_DICT["inv_vcoord_list"] = []
        GR_DICT["inv_edge_list"] = []

    for edge in GR_DICT["edge_list"]:
        vertices = []

        vertices.append(edge["vertices"][0])
        vertices.append(edge["vertices"][-1])

        if len(vertices) >= 3:
            vertices.append(edge["vertices"][len(vertices)//2])

        GR_DICT["vcoord_list"].extend([tuple(vertex) for vertex in vertices])
    
    gen_invalid_vertices_and_ntwrk()

def gen_invalid_vertices_and_ntwrk():
    """
    Function to generate the invalid vertices for the full network.
    """

    valid_vertices = set(GR_DICT["vcoord_list"])

    

    def get_direction(x, y, dir=-1):
        """
        Function to get direction in which there are no roads and hence  add invalid vertices.
        """
 
        if (x+pix_dist, y) not in valid_vertices:
            if dir == -1 or dir == 1:
                return 1, (x+pix_dist, y) 
        elif (x, y+pix_dist) not in valid_vertices:
            if dir == -1 or dir == 2:
                return 2, (x, y+pix_dist)
        elif (x-pix_dist, y) not in valid_vertices:
            if dir == -1 or dir == 3:
                return 3, (x-pix_dist, y)
        elif (x, y-pix_dist) not in valid_vertices:
            if dir == -1 or dir == 4:
                return 4, (x, y-pix_dist)

        return -1, []

    # Sampling the valid vertices where we are going to add invalid vertices. 
    # Vary this to vary number of samples augmented. 
    num_smps = int(len(GR_DICT["vcoord_list"])*inv_smp_rat)
    valid_samp_vs = random.sample(GR_DICT["vcoord_list"], num_smps)


    for v in valid_samp_vs:
        dir, v_next = get_direction(v[0], v[1])
        
        if dir == -1:
            continue

        _, v_next_next = get_direction(v_next[0], v_next[1], dir)

        # Add the first invalid vertex
        GR_DICT["inv_vcoord_list"].append(v_next)
        GR_DICT["inv_edge_list"].append((v, v_next))

        if v_next_next != []:
            # Add the second invalid vertex
            GR_DICT["inv_vcoord_list"].append(v_next_next)
            GR_DICT["inv_edge_list"].append((v_next, v_next_next))
        

def gen_samples_for_tile(img_name, xst=128, yst=128):
    """
    Generating and storing  valid/invalid samples for a tile. 
    """
    
    x_st = xst
    x_end = tile_size + x_st

    y_st = yst
    y_end = tile_size + y_st

    valid_vertices = set(GR_DICT["vcoord_list"]) 
        
    def crop_img(i, j):
        return IMG_DICT["p_sat_img"][i+pad_size//2:i+pad_size//2*3,j+ pad_size//2:j+pad_size//2*3,:]

    imgcod_vs_nodidx =  {}

    nodidx_vs_features = []

    # Indexing the image co-ordinates to node indices
    nodidx = 0
    for i in range(x_st, x_end):
        for j in range(y_st, y_end):

            if tuple([j, i]) in GR_DICT["vcoord_list"] or tuple([j, i]) in GR_DICT["inv_vcoord_list"]:
                imgcod_vs_nodidx[(i - x_st, j - y_st)] = nodidx
                nodidx+=1
                nodidx_vs_features.append( crop_img(i, j).flatten())
                print(f"Done with {nodidx}/{IMG_DICT['sat_img'].shape[0]*IMG_DICT['sat_img'].shape[1]} of node feature cropping of  {img_name}/144.")
        
    
    print("Finished with cropping the features for each pixel.")

    # By default all vertices are invalid
    nodey_vals = np.zeros(nodidx)
    

    
    # Creaitng the valid edge list for the GNN  Data structure. 
    edgev_list = []
    for edge in GR_DICT["edge_list"]:
        e_vertices = edge["vertices"]

        prev_e_vertex = -1

        for e_vertex in e_vertices:
            #imgcod_vs_isroad[e_vertex[1]][e_vertex[0]] = True
            if e_vertex[1] >= x_end or e_vertex[1] < x_st  or e_vertex[0] >= y_end or e_vertex[0] < y_st:
                continue
            
            if not (e_vertex[1] - x_st, e_vertex[0] - y_st) in imgcod_vs_nodidx:
                continue

            curr_ver = int(imgcod_vs_nodidx[(e_vertex[1] - x_st, e_vertex[0] - y_st)])

            nodey_vals[curr_ver] = 1

            if prev_e_vertex != -1:
                edgev_list.append([prev_e_vertex, curr_ver])
                edgev_list.append([curr_ver, prev_e_vertex])

            prev_e_vertex = curr_ver
    
    for edge in GR_DICT["inv_edge_list"]:

        v1 =  edge[0]
        v2 = edge[1]

        # Making sure it is a valid edge. 
        if v1[1] >= x_end or v1[1] < x_st  or v1[0] >= y_end or v1[0] < y_st:
            continue
        if v2[1] >= x_end or v2[1] < x_st  or v2[0] >= y_end or v2[0] < y_st:
            continue
        
        v1_ver = imgcod_vs_nodidx[(v1[1] - x_st, v1[0] - y_st)]
        v2_ver = imgcod_vs_nodidx[(v2[1] - x_st, v2[0] - y_st)]

        if v1 in valid_vertices:
            nodey_vals[v1_ver] = 1
        if v2 in valid_vertices:
            nodey_vals[v2_ver] = 1

        edgev_list.append([v1_ver, v2_ver])
        edgev_list.append([v2_ver, v1_ver])
    

    node_feature_tensor = torch.tensor(nodidx_vs_features, dtype=torch.float)
    if edgev_list == []:
        edge_index_tensor = torch.empty(2, 0, dtype=torch.long) 
    else:
        edge_index_tensor = torch.tensor(edgev_list, dtype=torch.long).t().contiguous()
    nodey_tensor = torch.tensor(nodey_vals == 1, dtype=torch.bool)


    data = Data(x=node_feature_tensor, edge_index=edge_index_tensor, y=nodey_tensor)
    print(data.validate())

    node_vs_img_cood = {}
    for k,v in imgcod_vs_nodidx.items():
        node_vs_img_cood[v] = k
    
    with open(f'{DATASET_STORE_PATH}/{img_name}_{x_st//tile_size}_{y_st//tile_size}_nodeid_vsimgcod.pkl', 'wb') as f:
        pickle.dump(node_vs_img_cood, f)
        
    torch.save(data, str(DATASET_STORE_PATH / f'{img_name}_{x_st//tile_size}_{y_st//tile_size}_{inv_smp_rat}_{tile_size}_gnn_dset.pt'))


def iter_over_tiles_of_image(img_name):
    for stx in range(0, IMG_DICT["sat_img"].shape[0], tile_size):
        for sty in range(0, IMG_DICT["sat_img"].shape[1], tile_size):
            gen_samples_for_tile(img_name, xst=stx, yst=sty)
             

def iter_over_imgs():
    DATASET_STORE_PATH.mkdir(exist_ok=True)
    
    with open('../dataset/data_split.json','r') as jf:
        imgs_list = json.load(jf)['train']
    
    for img_name in imgs_list:
        load_imgs_and_ev(img_name)
        iter_over_tiles_of_image(img_name)


iter_over_imgs()