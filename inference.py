from get_graph import get_graph
from merge_data import convert_dict_to_data
import torch
from torch_geometric.loader import DataLoader
from torch_geometric.nn import GCNConv
import torch.nn.functional as F

class GNN(torch.nn.Module):
    def __init__(self, in_channels, hidden_channels, out_channels):
        super(GNN, self).__init__()
        self.conv1 = GCNConv(in_channels, hidden_channels)
        self.conv2 = GCNConv(hidden_channels, out_channels)
        self.fc = torch.nn.Linear(out_channels, 1)  # Output is a single number

    def forward(self, data):
        x, edge_index = data.x, data.edge_index
        x = self.conv1(x, edge_index)
        x = F.relu(x)
        x = self.conv2(x, edge_index)
        x = F.relu(x)
        x = torch.mean(x, dim=0)  # Mean of node features
        x = self.fc(x)
        return x

def inference(aig_path):
    model_path = 'checkpoints/best_model.pth'
    graph = get_graph(aig_path)
    data = convert_dict_to_data(graph, 0)
    model = GNN(in_channels=2, hidden_channels=64, out_channels=32)
    model.load_state_dict(torch.load(model_path))
    model.eval()
    loader = DataLoader([data], batch_size=1)
    for data in loader:
        out = model(data)
        print(out)

if __name__ == "__main__":
    aig_path = '/home/zxz/course-project/ml_integrated_circuit_design/project/tmp_data/train_aig/adder_0000341036.aig'
    inference(aig_path)