import torch
from gnn_model import GCN
import torch.nn.functional as F
import torch.nn as nn
from  torch.nn import BCELoss as bceloss
from torch_geometric.loader import DataLoader
import random
from datetime import datetime
from pathlib import Path
from sklearn.metrics import confusion_matrix, ConfusionMatrixDisplay
import matplotlib.pyplot as plt
import numpy as np
import copy
import cv2 
import pickle
import logging
import os
import json


"""
The following lines contain the hyperparams using which we may tune the model.
"""
exp_name = f"gnn_train_dsetfrm_rngdet_wresnet_lossfncbce_ovsqsh_l2reg_1000"
#exp_name = "del_thsi"
seed_val = 42
num_epochs = 1000
lr = 0.00001
num_node_features = 3072
cl_0_wgt = 3
plot_epoch_start = 1
batch_size=1
n_train = 50
weight_decay=1e-2
use_resnet = True
use_bce_loss = True
use_fulladj = (batch_size == 1) & False
use_virtual_node = True

hyperparams = {
    "seed_val" : seed_val,
    "num_epochs": num_epochs,
    "lr": lr,
    "num_node_features": num_node_features,
    "cl_0_wgt": cl_0_wgt,
    "batch_size": batch_size,
    "n_train": n_train,
    "weight_decay": weight_decay, 
    "use_resnet": use_resnet,
    "use_bce_loss": use_bce_loss,
    "use_fulladj": use_fulladj,
    "use_virtual_node": use_virtual_node
}



# Seeding 
torch.manual_seed(seed_val)
random.seed(seed_val)

# Device Selection
device = torch.device('cuda:1' if torch.cuda.is_available() else 'cpu')

# Init the model. 
model = GCN(num_node_features=num_node_features, use_resnet=True, final_all_adj_layer=use_fulladj).to(device)
#model_state_dict = torch.load("/home/ramd_rm/workspace/RNGDetPlusPlus/GNN/Reports/gnn_best_model_train_50graph/gnn.pt")
#model.load_state_dict(model_state_dict, strict=True)
optimizer = torch.optim.Adam(model.parameters(), lr=lr, weight_decay=weight_decay)
model.train()
criterion = F.nll_loss

criterion_bce = nn.BCELoss()

#data_folder_path = Path("GNN_Dset_3vertex32x32")
data_folder_path = Path("Dataset_Frm_Rngdet/Sample_GNN_Dset_frm_RNGDet")


# Output folders to write out the results. 
now = datetime.now()
#date and time format: dd/mm/YYYY H:M:S
format = "%d/%m/%Y %H:%M:%S"
#format datetime using strftime() 
time1 = now.strftime(format)
reports_folder_path = Path("Reports") / f"{exp_name}_{now}"
reports_folder_path.mkdir(parents=True, exist_ok=True)

with open(reports_folder_path / "hyperparams.json", "w") as json_file:
    json_file.write(json.dumps(hyperparams, indent=2))

# Logging handlers to write loss values. 
logger = logging.getLogger()
handler = logging.FileHandler(str(reports_folder_path / 'loss.log'))
logger.addHandler(handler)
logger.setLevel(logging.INFO)


# Loading in the datasets. 
all_fls = os.listdir(data_folder_path)
all_pkl_fls = [fl for fl in all_fls if fl.endswith(".pkl")]
all_pt_fls = [f"{fl[0:fl.find('.')]}.pt" for fl in all_pkl_fls if "r" not in fl]

with open('data_split.json','r') as jf:
    spl_json = json.load(jf)
    train_list = spl_json['train']
    test_list = spl_json['test']

test_f_list = [f"{idx}.pt" for idx in test_list if f"{idx}.pt" in all_pt_fls]
train_f_list = [f"{idx}.pt" for idx in train_list if f"{idx}.pt" in all_pt_fls]
all_f_list = all_pt_fls

train_dset_list = []
test_dset_list = []


def add_virtual_node_edge(dset):
    dset.x = torch.cat((dset.x, torch.zeros((1, dset.x.shape[1]))), dim=0)
    vnode = dset.x.shape[0] - 1

    lis1 = [i for i in range(vnode)]
    lis2 = [vnode for _ in range(vnode)]

    vedge_index = torch.Tensor([lis1, lis2]).to(torch.int64)

    dset.edge_index = torch.cat((dset.edge_index, vedge_index), dim=1).to(torch.int64)
    dset.y = torch.cat((dset.y, torch.zeros(1)), dim=0).to(torch.int64)

for datafpath in data_folder_path.iterdir():
    fname = datafpath.parts[-1]
    if  fname  in  train_f_list:
        train_dset_list.append(torch.load(datafpath))
        if use_virtual_node:
            add_virtual_node_edge(train_dset_list[-1])
        train_dset_list[-1] = train_dset_list[-1].to(device)
    elif fname in test_f_list:
        test_dset_list.append(torch.load(datafpath))
        if use_virtual_node:
            add_virtual_node_edge(test_dset_list[-1])
        test_dset_list[-1] = test_dset_list[-1].to(device)



train_loader = DataLoader(train_dset_list, batch_size=batch_size, shuffle=False)
test_loader = DataLoader(test_dset_list, batch_size=batch_size, shuffle=False)


# PLotting the  train/test loss curves
def  plot_train_test_metric_vs_epochs(train_losses=[], test_losses=[], metric="loss"):

    epochs = [epoch for epoch in range(plot_epoch_start, 
                num_epochs)]

    plt.close('all')
    fig, ax = plt.subplots(tight_layout=True)

    p1,  = ax.plot(epochs, train_losses,  color='blue', marker='o', label='Train ' + metric)
    p2,  = ax.plot(epochs, test_losses,  color='red', marker='o', label='Test ' + metric)

    plt_title = 'Loss Plot'
    ax.set_xlabel("Epoch Count")
    ax.set_ylabel(metric)
    ax.set_title(plt_title)

    ax.legend(handles=[p1, p2], title='Legend', bbox_to_anchor=(1.05, 1), loc='upper left')
    ax.grid(True)

    plt.show()
    
    fig.savefig(str(reports_folder_path / "loss_plot.jpg"))
    plt.close('all')

# Method to train the classification on the nodes. 
def train():
    train_losses = []
    test_losses = []
    
    model.train()
    with torch.set_grad_enabled(True):

        for epoch in range(num_epochs):
            tre_loss = []
            tse_loss = []

            for data in train_loader:            

                if data.y.shape[0] == 0:
                    continue
                
                data = data.to(device)

                optimizer.zero_grad()
                out = model(data)

                #loss = criterion(out, data.y.to(torch.int64), reduction="sum", weight=torch.Tensor([cl_0_wgt, 1]).to(device))
                loss = criterion_bce(out.to(torch.float64), data.y.to(torch.double))

                tre_loss.append(loss.item())

                loss.backward()
                optimizer.step()
        
            for datacp in test_loader:
                data = datacp.to(device)
                out = model(data)
                loss = criterion_bce(out.to(torch.float64), data.y.to(torch.float64))
                tse_loss.append(loss.item())

            if epoch > plot_epoch_start -1:
                train_losses.append(np.sum(tre_loss)/len(train_f_list))
                test_losses.append(np.sum(tse_loss)/len(test_f_list))
            
            if epoch%10 == 0:
                print(f"The loss  for epoch: {epoch}/{num_epochs} is {np.sum(tre_loss)/len(train_f_list)}")
                logger.info(f"The loss  for epoch: {epoch}/{num_epochs} is {np.sum(tre_loss)/len(train_f_list)}")

    
    plot_train_test_metric_vs_epochs(train_losses, test_losses)

def inference():

    cm = None
    
    model.eval()
    with torch.set_grad_enabled(False):

        for datafpath in data_folder_path.iterdir():

            if not datafpath.parts[-1] in all_f_list:
                continue

            data = torch.load(datafpath).to(device)

            if data.y.shape[0] == 0:
                continue
                
            image_name = datafpath.parts[-1][0:datafpath.parts[-1].find(".pt")]
                    
            sat_img = cv2.imread(f"../dataset/20cities/region_{image_name}_sat.png")

            node_idx_fpath =  data_folder_path / f'{image_name}.pkl'
            unpickd_file = open(node_idx_fpath, 'rb')
            nodeidx_vs_imgcod = pickle.load(unpickd_file, encoding='bytes')

            out = model(data)

            loss = criterion_bce(out.to(torch.float64), data.y.to(torch.float64))

            print(f"The loss on image name: {image_name} for inference phase  is {loss.item()}")
            
            #cm_new = confusion_matrix(data.y.to("cpu").to(torch.int32), torch.argmax(out, dim=1).to("cpu").to(torch.int32), labels=[0, 1])
            cm_new = confusion_matrix(data.y.to("cpu").to(torch.int32), (out > 0.5).to("cpu").to(torch.int32), labels=[0, 1])

            if cm is not None:
                cm += cm_new
            else:
                cm = cm_new

            # Plotting the ground truth. 
            inv_ids = (data.y.to("cpu").to(torch.int32) == 0).nonzero(as_tuple=False).tolist()
            vids = (data.y.to("cpu").to(torch.int32) == 1).nonzero(as_tuple=False).tolist()

            nsatimg = copy.deepcopy(sat_img)
            for inv_id in inv_ids:
                nsatimg = cv2.circle(nsatimg, (nodeidx_vs_imgcod[inv_id[0]][0], nodeidx_vs_imgcod[inv_id[0]][1]), 3, (255,0,255), 1)
                nsatimg = cv2.circle(nsatimg, (nodeidx_vs_imgcod[inv_id[0]][0], nodeidx_vs_imgcod[inv_id[0]][1]), 3, (0,0,255), 3)

            for inv_id in vids:
                nsatimg = cv2.circle(nsatimg, (nodeidx_vs_imgcod[inv_id[0]][0], nodeidx_vs_imgcod[inv_id[0]][1]), 3, (255,0,255), 1)

            edge_lis =  data.edge_index.tolist()
            
            for idx  in range(len(edge_lis[0])):

                v1 = edge_lis[0][idx]
                v2 = edge_lis[1][idx]

                nsatimg = cv2.line(nsatimg, (nodeidx_vs_imgcod[v1][0], nodeidx_vs_imgcod[v1][1]), (nodeidx_vs_imgcod[v2][0], nodeidx_vs_imgcod[v2][1]), (227,188, 0), 2)

            
            cv2.imwrite(str(reports_folder_path / f"{image_name}_invver_gt.png"),nsatimg)

            # Plotting the predictions. 
            #o1 = torch.argmax(out, dim=1)
            o1 = (out > 0.5).to("cpu").to(torch.int32)
            inv_ids = (o1.to("cpu").to(torch.int32) == 0).nonzero(as_tuple=False).tolist()
            vids = (o1.to("cpu").to(torch.int32) == 1).nonzero(as_tuple=False).tolist()

            nsatimg = copy.deepcopy(sat_img)

            for inv_id in inv_ids:
                nsatimg = cv2.circle(nsatimg, (nodeidx_vs_imgcod[inv_id[0]][0], nodeidx_vs_imgcod[inv_id[0]][1]), 3, (255,0,255), 1)
                nsatimg = cv2.circle(nsatimg, (nodeidx_vs_imgcod[inv_id[0]][0], nodeidx_vs_imgcod[inv_id[0]][1]), 3, (0,0,255), 3)

            for inv_id in vids:
                nsatimg = cv2.circle(nsatimg, (nodeidx_vs_imgcod[inv_id[0]][0], nodeidx_vs_imgcod[inv_id[0]][1]), 3, (255,0,255), 1)

            edge_lis =  data.edge_index.tolist()
            
            for idx  in range(len(edge_lis[0])):

                v1 = edge_lis[0][idx]
                v2 = edge_lis[1][idx]

                nsatimg = cv2.line(nsatimg, (nodeidx_vs_imgcod[v1][0], nodeidx_vs_imgcod[v1][1]), (nodeidx_vs_imgcod[v2][0], nodeidx_vs_imgcod[v2][1]), (227,188, 0), 2)

            
            cv2.imwrite(str(reports_folder_path / f"{image_name}_invver_pred.png"),nsatimg)

            # Plotting Confusion matrix of individual image
            disp = ConfusionMatrixDisplay(confusion_matrix=cm_new)
            disp.plot()
            plt.show()
            plt.savefig(reports_folder_path / f'conf_matrix_{image_name}.png')


    # Plotting Confusion matrix of all image
    disp = ConfusionMatrixDisplay(confusion_matrix=cm)
    disp.plot()
    plt.show()
    plt.savefig(reports_folder_path / 'conf_matrix_all.png')
    torch.save(model.to("cpu").state_dict(), reports_folder_path / f"gnn.pt")

# Required for cuda_NN bug
def force_cudnn_initialization():
    s = 1
    torch.nn.functional.conv2d(torch.zeros(s, s, s, s, device=device), torch.zeros(s, s, s, s, device=device))

force_cudnn_initialization()
train()
inference() 