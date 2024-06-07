import torch
from torch_geometric.loader import DataLoader
from torch_geometric.nn import GCNConv
import torch.nn.functional as F
import pickle
from torch_geometric.data import Data, InMemoryDataset
from torch_geometric.nn import BatchNorm

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')


# class GNN(torch.nn.Module):
#     def __init__(self, in_channels, hidden_channels, out_channels):
#         super(GNN, self).__init__()
#         self.conv1 = GCNConv(in_channels, hidden_channels)
#         self.bn1 = BatchNorm(hidden_channels)
#         self.conv2 = GCNConv(hidden_channels, hidden_channels)
#         self.bn2 = BatchNorm(hidden_channels)
#         self.conv3 = GCNConv(hidden_channels, out_channels)
#         self.bn3 = BatchNorm(out_channels)
#         self.fc1 = torch.nn.Linear(out_channels, 128)
#         self.fc2 = torch.nn.Linear(128, 64)
#         self.fc3 = torch.nn.Linear(64, 32)
#         self.fc4 = torch.nn.Linear(32, 1)  # Output is a single number
#         self.dropout = torch.nn.Dropout(p=0.5)  # Dropout layer

#     def forward(self, data):
#         x, edge_index = data.x, data.edge_index
#         x = self.conv1(x, edge_index)
#         x = self.bn1(x)
#         x = F.relu(x)
#         x = self.conv2(x, edge_index)
#         x = self.bn2(x)
#         x = F.relu(x)
#         x = self.conv3(x, edge_index)
#         x = self.bn3(x)
#         x = F.relu(x)
#         x = torch.mean(x, dim=0)  # Mean of node features
#         x = self.fc1(x)
#         x = F.relu(x)
#         x = self.dropout(x)  # Apply dropout
#         x = self.fc2(x)
#         x = F.relu(x)
#         x = self.fc3(x)
#         x = F.relu(x)
#         x = self.fc4(x)
#         return x

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


        
def train(model, train_loader, optimizer, criterion, device):
    model.train()
    total_loss = 0
    for data in train_loader:
        data = data.to(device)
        optimizer.zero_grad()
        out = model(data)
        loss = criterion(out, data.y)
        loss.backward()
        optimizer.step()
        total_loss += loss.item()
    return total_loss / len(train_loader)

def evaluate(model, val_loader, criterion, device):
    model.eval()
    total_loss = 0
    with torch.no_grad():
        for data in val_loader:
            data = data.to(device)
            out = model(data)
            loss = criterion(out, data.y)
            total_loss += loss.item()
    return total_loss / len(val_loader)

def train_task1():
    # Load merged graph data and labels
    train_dataset = torch.load('data/train_dataset.pt')
    val_dataset = torch.load('data/val_dataset.pt')
    batch_size = 1
    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
    val_loader = DataLoader(val_dataset, batch_size=batch_size, shuffle=False)

    model = GNN(in_channels=2, hidden_channels=64, out_channels=32).to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=0.001)
    criterion = torch.nn.MSELoss()

    num_epochs = 10
    best_val_loss = float('inf')  # 初始化最佳验证损失为无穷大
    best_model_path = 'checkpoints/best_model.pth'  # 保存最佳模型的路径

    for epoch in range(num_epochs):
        train_loss = train(model, train_loader, optimizer, criterion, device)
        val_loss = evaluate(model, val_loader, criterion, device)
        print(f'Epoch {epoch+1}, Train Loss: {train_loss:.4f}, Validation Loss: {val_loss:.4f}')
        # 如果当前验证损失低于最佳验证损失，则更新最佳验证损失并保存模型
        if val_loss < best_val_loss:
            best_val_loss = val_loss
            torch.save(model.state_dict(), best_model_path)
            print(f'Saving best model with validation loss: {val_loss:.4f}')

    # Print predictions for a batch in validation set
    model.eval()
    with torch.no_grad():
        total_loss = 0
        for data in val_loader:
            data = data.to(device)
            out = model(data)
            loss = criterion(out, data.y)
            total_loss += loss.item()
        print(f'Validation Loss: {total_loss/len(val_loader)}')
            # print(f'Predicted: {out.item()}, Actual: {data.y.item()}')

def train_task2():
    # Load merged graph data and labels
    train_dataset = torch.load('data/train_dataset_task2.pt')
    val_dataset = torch.load('data/val_dataset_task2.pt')
    batch_size = 1
    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
    val_loader = DataLoader(val_dataset, batch_size=batch_size, shuffle=False)

    model = GNN(in_channels=2, hidden_channels=64, out_channels=32).to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=0.001)
    criterion = torch.nn.MSELoss()

    num_epochs = 20
    best_val_loss = float('inf')  # 初始化最佳验证损失为无穷大
    best_model_path = 'checkpoints/best_model_task2.pth'  # 保存最佳模型的路径

    for epoch in range(num_epochs):
        train_loss = train(model, train_loader, optimizer, criterion, device)
        val_loss = evaluate(model, val_loader, criterion, device)
        print(f'Epoch {epoch+1}, Train Loss: {train_loss:.4f}, Validation Loss: {val_loss:.4f}')
        # 如果当前验证损失低于最佳验证损失，则更新最佳验证损失并保存模型
        if val_loss < best_val_loss:
            best_val_loss = val_loss
            torch.save(model.state_dict(), best_model_path)
            print(f'Saving best model with validation loss: {val_loss:.4f}')

    # Print predictions for a batch in validation set
    model.eval()
    with torch.no_grad():
        total_loss = 0
        for data in val_loader:
            data = data.to(device)
            out = model(data)
            loss = criterion(out, data.y)
            total_loss += loss.item()
        print(f'Validation Loss: {total_loss/len(val_loader)}')
            # print(f'Predicted: {out.item()}, Actual: {data.y.item()}')

if __name__ == "__main__":
    train_task2()