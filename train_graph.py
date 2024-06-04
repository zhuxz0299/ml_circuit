import torch
from torch_geometric.loader import DataLoader
from torch_geometric.nn import GCNConv
import torch.nn.functional as F
import pickle
from torch_geometric.data import Data, InMemoryDataset

# Define CustomDataset class again
class CustomDataset(InMemoryDataset):
    def __init__(self, root, data_list, transform=None, pre_transform=None):
        self.data_list = data_list
        super(CustomDataset, self).__init__(root, transform, pre_transform)
        self.data, self.slices = self.collate(data_list)

    @property
    def raw_file_names(self):
        return []

    @property
    def processed_file_names(self):
        return []

    def download(self):
        pass

    def process(self):
        pass

# Load merged graph data and labels
dataset = torch.load('data/merged_graph_data.pt')
with open('data/merged_labels.pkl', 'rb') as f:
    labels = pickle.load(f)

# Create data loader
train_loader = DataLoader(dataset, batch_size=32, shuffle=True)

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

model = GNN(in_channels=2, hidden_channels=64, out_channels=32)
optimizer = torch.optim.Adam(model.parameters(), lr=0.01)
criterion = torch.nn.MSELoss()

model.train()
for epoch in range(20):
    total_loss = 0
    for i, data in enumerate(train_loader):
        optimizer.zero_grad()
        out = model(data)
        target = torch.tensor([labels[i]], dtype=torch.float32)
        loss = criterion(out, target)
        loss.backward()
        optimizer.step()
        total_loss += loss.item()
    print(f'Epoch {epoch+1}, Loss: {total_loss/len(train_loader)}')