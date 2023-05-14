from gnn_model import GCN
import cv2
import matplotlib.pyplot as plt
import torch.nn.functional as F
import torch.nn as nn
import torch
import pickle
import numpy as np
from PIL import Image
import os
import json
import copy
import pandas as pd
from pathlib import Path
from scipy.spatial import cKDTree


from torch_geometric.data import Data
from PIL import Image, ImageDraw
from datetime import datetime


num_node_features = 3072
only_test = True
op_path = "Results_Correction/gnn_coprr_train_wresnet30_lssbce_ovsqsh_l2reg_200"

# Output folders to write out the results. 
now = datetime.now()
op_path =  Path(f"{op_path}_{now}")
op_path.mkdir(parents=True, exist_ok=True)


device = torch.device('cuda:7' if torch.cuda.is_available() else 'cpu')

# Paths of Roadtracer images and results
all_roadtracer_imgs = os.listdir("/home/ramd_rm/workspace/roadtracer/dataset/data/imagery")
all_result_grps = os.listdir("/home/ramd_rm/workspace/RNGDetPlusPlus/RNGDet_multi_ins/test/json")
rel_img_pths = [img[0:img.find('_sat')] for img in all_roadtracer_imgs if f"{img}.json" in all_result_grps]
all_roadtracer_imgs = [f"/home/ramd_rm/workspace/roadtracer/dataset/data/imagery/{imgpth}_sat.png" for imgpth in rel_img_pths]
all_result_grps = [f"/home/ramd_rm/workspace/RNGDetPlusPlus/RNGDet_multi_ins/test/json/{imgpth}_sat.png.json" for imgpth in rel_img_pths]


# Paths of all Sat2graph images and results
with open('data_split.json','r') as jf:
    spl_json = json.load(jf)
    sat2graph_trainnums = spl_json['train']
    sat2graph_testnums = spl_json['test']
    sat2graph_nums = sat2graph_trainnums + sat2graph_testnums
    

if only_test:
    sat2graph_nums = sat2graph_testnums

s2gimgs = os.listdir("/home/ramd_rm/workspace/RNGDetPlusPlus/RNGDet_multi_ins/test/json/")
s2gimgs = [img for img in s2gimgs if not "sat" in img and not "_" in img]
s2gimgs = [int(img[0:img.find(".json")])  for img in s2gimgs]

all_sat2grp_grps = [f"/home/ramd_rm/workspace/RNGDetPlusPlus/RNGDet_multi_ins/test/json/{imgnum}.json" for imgnum in sat2graph_nums]
all_sat2graph_imgs = [f"/home/ramd_rm/workspace/RNGDetPlusPlus/dataset/20cities/region_{imgnum}_sat.png" for imgnum in sat2graph_nums]
all_pt_fls = [f"{idx}.pt" for idx in sat2graph_nums]
all_nodeidx_vs_imgcod_pkl_fls = [f"{idx}.pkl" for idx in sat2graph_nums]

# Current results and graphs
data_folder_path = Path(f"Dataset_Frm_Rngdet/GNN_testDset_frm_RNGDet{now}")
data_folder_path.mkdir(parents=True, exist_ok=True)
data_folder_path = Path("Dataset_Frm_Rngdet/Sample_GNN_Dset_frm_RNGDet")


# Initializing the model
model = GCN(num_node_features=num_node_features, use_resnet=True)
model_state_dict = torch.load("/home/ramd_rm/workspace/RNGDetPlusPlus/GNN/Reports/gnn_train_dsetfrm_rngdet_wresnet_lossfncbce_ovsqsh_l2reg_200_2023-03-14_09:03:45.230373/gnn.pt", map_location='cpu')
model.load_state_dict(model_state_dict, strict=True)
model = model.to(device)

results_dict = {}

def store_refined_graph(edge_list, img_cod_vsnodeidx, is_valid, img_name):
    '''
    {
        (edge[i][1], edge[i][0]) : [neighbours of this node in the same format ] 
    }
    '''
    
    g_dict = {}

    for edge in edge_list:
        v1 = img_cod_vsnodeidx[tuple(edge[0])]
        v2 = img_cod_vsnodeidx[tuple(edge[1])]

        v1_rev = tuple((edge[0][1], edge[0][0]))
        v2_rev = tuple((edge[1][1], edge[1][0]))


        if not is_valid[v1] or not is_valid[v2]:
            continue
        
        if v1_rev not in g_dict:
            g_dict[v1_rev] = []

        if v2_rev not in g_dict:
            g_dict[v2_rev] = []
        
        if v1_rev not in set(g_dict[v2_rev]):
            g_dict[v2_rev].append(v1_rev)
        
        if v2_rev not in set(g_dict[v1_rev]):
            g_dict[v1_rev].append(v2_rev)
        
    f = open(op_path/f'{img_name}.p','wb')
    pickle.dump(g_dict,f)

# Use this to construct the binary map required for calculation of pixel wise metric.
def update_historical_map(src,dst, historical_map):
    '''
    Update the historical map by adding a line starting from src to dst.

    :param src (list, length=2): src point of the added line
    :param dst (list, length=2): src point of the added line
    '''
    src = np.array(src)
    dst = np.array(dst)

    p = src
    d = dst - src
    N = np.max(np.abs(d))

    historical_map[src[1],src[0]] = 255
    historical_map[dst[1],dst[0]] = 255

    if N:
        s = d / (N)
        for i in range(0,N):
            p = p + s
            historical_map[int(round(p[1])),int(round(p[0]))] = 255


# Function used to generate the binary mask from Edge list. 
def gen_mask_from_edge_list(edge_list, img_cod_vsnodeidx=None, is_valid=None):

    historical_map = np.zeros((2048,2048))
    for edge in edge_list:
        if is_valid:
            v1 = img_cod_vsnodeidx[tuple(edge[0])]
            v2 = img_cod_vsnodeidx[tuple(edge[1])]

            if not is_valid[v1] or not is_valid[v2]:
                continue

        update_historical_map(edge[0], edge[1], historical_map)
    
    return historical_map

    #pred_graph = np.asarray(Image.fromarray(historical_map.astype(np.uint8)).convert('RGB'))
    #return pred_graph


# Function to calculate the metrics
def calculate_scores(gt_points,pred_points, metric_px_thr):
    gt_tree = cKDTree(gt_points)
    if len(pred_points):
        pred_tree = cKDTree(pred_points)
    else:
        return 0,0,0
    dis_gt2pred,_ = pred_tree.query(gt_points, k=1)
    dis_pred2gt,_ = gt_tree.query(pred_points, k=1)
    recall = len([x for x in dis_gt2pred if x<metric_px_thr])/len(dis_gt2pred)
    pre = len([x for x in dis_pred2gt if x<metric_px_thr])/len(dis_pred2gt)
    r_f = 0
    if pre*recall:
        r_f = (2*recall * pre) / (pre+recall)
    return pre, recall, r_f

def pixel_eval_metric(pred_mask,gt_mask, metric_px_thr=3):
    def tuple2list(t):
        return [[t[0][x],t[1][x]] for x in range(len(t[0]))]

    gt_points = tuple2list(np.where(gt_mask!=0))
    pred_points = tuple2list(np.where(pred_mask!=0))

    return calculate_scores(gt_points,pred_points, metric_px_thr)

# Visualizing the classified graph nodes. 
def visualize_graph(img, edge_list, img_cod_vsnodeidx, is_valid, img_name):
    nsatimg = copy.deepcopy(img)

    for edge in edge_list:

        v1 = img_cod_vsnodeidx[tuple(edge[0])]
        v2 = img_cod_vsnodeidx[tuple(edge[1])]


        if is_valid[v1] == 1:
            nsatimg = cv2.circle(nsatimg, (edge[0][0], edge[0][1]), 2,(255,0,255), 1)
        else:
            nsatimg = cv2.circle(nsatimg, (edge[0][0], edge[0][1]), 2, (255,0,255), 1)
            nsatimg = cv2.circle(nsatimg, (edge[0][0], edge[0][1]), 2, (255, 0,0), 3)

        if is_valid[v2] == 1:
            nsatimg = cv2.circle(nsatimg,  (edge[1][0], edge[1][1]), 2, (255,0,255), 1)
        else:
            nsatimg = cv2.circle(nsatimg,  (edge[1][0], edge[1][1]), 2, (255,0, 255), 1)
            nsatimg = cv2.circle(nsatimg,  (edge[1][0], edge[1][1]), 2, (255, 0,0), 3)
            

        pt1 = (edge[0][0], edge[0][1])
        pt2 = (edge[1][0], edge[1][1])

        nsatimg = cv2.line(nsatimg, pt1, pt2, (0,188, 227), 2)
    
    Image.fromarray(nsatimg.astype(np.uint8)).convert('RGB').save(f'/home/ramd_rm/workspace/RNGDetPlusPlus/GNN/{op_path}/{img_name}.png')

'''
Function to construct GNN Data class from 
1. img : The image for the features
2. edge_list_coord: The list of edges 
3. gt_mask: Binary array of ground truth of where it is ground truth and where it is not. 
'''
def construct_gnn_data(img, edge_list_coord, gt_mask):
    
    img_cod_vsnodeidx = {}
    node_vs_img_cood = {}

    pad_size = 32
    node_cnt = -1
    pd_img = np.pad(img,np.array(((pad_size,pad_size),(pad_size,pad_size),(0,0))),'constant')
    edge_list = []
    feature_list = []
    labels_list = []

    def tuple2list(t):
        return [[t[0][x],t[1][x]] for x in range(len(t[0]))]

    gt_points = tuple2list(np.where(gt_mask!=0))
    gt_tree = cKDTree(gt_points)
    gt_thr = 30 # Tolerance distance till which point is considered a valid point. 


    def crop_img(i, j):
        return pd_img[i+pad_size//2:i+pad_size//2*3,j+ pad_size//2:j+pad_size//2*3,:]


    for edge in edge_list_coord:
        v1 = edge[0]
        v2 = edge[1]

        if tuple(v1) not in img_cod_vsnodeidx:
            node_cnt += 1
            img_cod_vsnodeidx[tuple(v1)] = node_cnt
            node_vs_img_cood[node_cnt] = tuple(v1)
            feature_list.append(crop_img(v1[0], v1[1]).flatten())

            dis_pred2gt,_ = gt_tree.query((v1[1], v1[0]), k=1)
            res = int(dis_pred2gt < gt_thr)
            labels_list.append(res)


        if tuple(v2) not in img_cod_vsnodeidx:
            node_cnt += 1
            img_cod_vsnodeidx[tuple(v2)] = node_cnt
            node_vs_img_cood[node_cnt] = tuple(v2)
            feature_list.append(crop_img(v2[0], v2[1]).flatten())

            dis_pred2gt,_ = gt_tree.query((v2[1], v2[0]), k=1)
            res = int(dis_pred2gt < gt_thr)
            labels_list.append(res)


        edge_list.append([img_cod_vsnodeidx[tuple(v1)], img_cod_vsnodeidx[tuple(v2)]])
        edge_list.append([img_cod_vsnodeidx[tuple(v2)], img_cod_vsnodeidx[tuple(v1)]])

    node_feature_tensor = torch.tensor(feature_list, dtype=torch.float)
    edge_index_tensor = torch.tensor(edge_list, dtype=torch.long).t().contiguous()
    labels_tensor = torch.tensor(labels_list, dtype=torch.long)

    data = Data(x=node_feature_tensor, edge_index=edge_index_tensor, y=labels_tensor)
    
    return data, img_cod_vsnodeidx, node_vs_img_cood


# Method to directly load in dataset in case already stored. 
def load_gnn_data(img_idx):
    data = torch.load(data_folder_path / f"{img_idx}.pt")
    
    node_idx_fpath =  data_folder_path / f'{img_idx}.pkl'
    unpickd_file = open(node_idx_fpath, 'rb')
    nodeidx_vs_imgcod = pickle.load(unpickd_file, encoding='bytes')

    node_idx_fpath =  data_folder_path / f'{img_idx}_r.pkl'
    unpickd_file = open(node_idx_fpath, 'rb')
    img_cod_vsnodeidx = pickle.load(unpickd_file, encoding='bytes')


    return data, img_cod_vsnodeidx, nodeidx_vs_imgcod


for idx in range(len(all_sat2graph_imgs)):

    img = np.array(Image.open(os.path.join(all_sat2graph_imgs[idx])).resize((2048, 2048)))

    with open(all_sat2grp_grps[idx],'r') as jf:
        edge_list = json.load(jf)


    gt_graph = cv2.imread(f"/home/ramd_rm/workspace/RNGDetPlusPlus/dataset/segment/{sat2graph_nums[idx]}.png")
    #gnn_dset, img_cod_vsnodeidx, node_vs_img_cood = construct_gnn_data(img, edge_list, gt_graph)
    gnn_dset, img_cod_vsnodeidx, node_vs_img_cood = load_gnn_data(sat2graph_testnums[idx])

    img_name = all_sat2graph_imgs[idx][all_sat2graph_imgs[idx].rfind("/")+1:all_sat2graph_imgs[idx].rfind("_sat")]

    if gnn_dset.x.shape[0] == 0:
        continue
    
    # Store Dataset in case not already stored
    '''torch.save(gnn_dset, f"{data_folder_path}/{sat2graph_nums[idx]}.pt")
    with open(f"{data_folder_path}/{sat2graph_nums[idx]}.pkl", 'wb') as f:
        pickle.dump(node_vs_img_cood, f)
    with open(f"{data_folder_path}/{sat2graph_nums[idx]}_r.pkl", 'wb') as f:
        pickle.dump(img_cod_vsnodeidx, f)'''



    print(gnn_dset)
    print(img_name)

    is_valid = (model(gnn_dset.to("cuda:7")) > 0.5).to(torch.int32).detach().tolist()

    
    edge_list = gnn_dset.edge_index.tolist()
    edge_list_nod = [[edge_list[0][idx], edge_list[1][idx]] for idx in  range(0, len(edge_list[0]), 1)]
    edge_list_cor = []

    for edge in edge_list_nod:
        edge_list_cor.append([node_vs_img_cood[edge[0]], node_vs_img_cood[edge[1]]])

    visualize_graph(img, edge_list_cor, img_cod_vsnodeidx, is_valid, f"{img_name}_pred")
    visualize_graph(img, edge_list_cor, img_cod_vsnodeidx, gnn_dset.y.detach().tolist(), f"{img_name}_gt")


    pred_graph = gen_mask_from_edge_list(edge_list_cor)
    pred_corr_graph = gen_mask_from_edge_list(edge_list_cor, img_cod_vsnodeidx, is_valid)
    
    store_refined_graph(edge_list_cor, img_cod_vsnodeidx, is_valid, img_name)

    results_dict[idx] = {"original" : {}, "corrected": {}}


    for metric_px_thr in [2 ,5 ,10]:
        og_prec , og_re, og_fsc = pixel_eval_metric(pred_graph, gt_graph, metric_px_thr)
        cor_prec , cor_re, cor_re = pixel_eval_metric(pred_corr_graph, gt_graph, metric_px_thr)
        
        results_dict[idx]["original"][metric_px_thr] = {}
        results_dict[idx]["corrected"][metric_px_thr] = {}


        results_dict[idx]["original"][metric_px_thr]["prec"] = og_prec
        results_dict[idx]["original"][metric_px_thr]["rec"] = og_re
        results_dict[idx]["original"][metric_px_thr]["f1_sc"] = og_fsc

        results_dict[idx]["corrected"][metric_px_thr]["prec"] = cor_prec
        results_dict[idx]["corrected"][metric_px_thr]["rec"] = cor_re
        results_dict[idx]["corrected"][metric_px_thr]["f1_sc"] = cor_re


    

results_list = []

for idx in results_dict:
    for typ in results_dict[idx]:
        for metric_px_thr in results_dict[idx][typ]:
            for met in results_dict[idx][typ][metric_px_thr]:
                results_list.append([idx, typ, metric_px_thr, met, results_dict[idx][typ][metric_px_thr][met]])

df = pd.DataFrame(results_list, columns=["img_name", "method", "pix_thresh", "metric", "value"])


with open(f"{op_path}/log_loss.txt", "w") as f:
    for metric_px_thr in [2 ,5 ,10]:
        f.write(f'For {metric_px_thr} px thresh The original precision is: {df[(df["method"] == "original") & (df["metric"] == "prec") & (df["pix_thresh"] == metric_px_thr)]["value"].mean()}\n')
        f.write(f'For {metric_px_thr} px thresh The corrected precision is: {df[(df["method"] == "corrected") & (df["metric"] == "prec") & (df["pix_thresh"] == metric_px_thr)]["value"].mean()}\n')


        f.write(f'For {metric_px_thr} px thresh The original recall is: {df[(df["method"] == "original") & (df["metric"] == "rec") & (df["pix_thresh"] == metric_px_thr)]["value"].mean()}\n')
        f.write(f'For {metric_px_thr} px thresh The corrected recall is: {df[(df["method"] == "corrected") & (df["metric"] == "rec") & (df["pix_thresh"] == metric_px_thr)]["value"].mean()}\n')

        f.write(f'For {metric_px_thr} px thresh The original f1score is: {df[(df["method"] == "original") & (df["metric"] == "f1_sc") & (df["pix_thresh"] == metric_px_thr)]["value"].mean()}\n')
        f.write(f'For {metric_px_thr} px thresh The corrected f1score is: {df[(df["method"] == "corrected") & (df["metric"] == "f1_sc") & (df["pix_thresh"] == metric_px_thr)]["value"].mean()}\n')
