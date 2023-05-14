import torch
import torch.nn as nn
import torch.nn.functional as F
from torch_geometric.nn import GCNConv

class GCN(torch.nn.Module):
    def __init__(self, num_node_features=48, num_classes=2, thr=0.5, use_resnet=False, final_all_adj_layer=False):
        super().__init__()

        self.use_resnet = use_resnet

        # Resnet ouitputs 1000 features for these 32*32*3 crops. 
        if self.use_resnet:
            self.resnet = torch.hub.load('pytorch/vision:v0.10.0', 'resnet18', pretrained=True)
            num_node_features = 1000

        self.conv1 = GCNConv(num_node_features, 16)
        self.conv2 = GCNConv(16, 16)
        self.conv3 = GCNConv(16, 16)
        self.conv4 = GCNConv(16, 16)

        self.fc1 = nn.Linear(16, 1)
        self.sigmoid = nn.Sigmoid()
        self.final_all_adj_layer = final_all_adj_layer


    def forward(self, data):
        x, edge_index = data.x, data.edge_index


        if self.use_resnet:
            x = self.resnet(torch.stack([x.view(3, 32, 32) for x in data.x]))

        x = self.conv1(x, edge_index)
        x = F.relu(x)

        x = self.conv2(x, edge_index)
        x = F.relu(x)

        x = self.conv3(x, edge_index)
        x = F.relu(x)

        if self.final_all_adj_layer:
            num_nodes = x.shape[0]
            edge_index = torch.ones(num_nodes, num_nodes).nonzero().t().to(x.device)

        x = self.conv4(x, edge_index)
        x = self.fc1(x).flatten()

        #res = F.log_softmax(x, dim=1)
        res = self.sigmoid(x)

        return res 