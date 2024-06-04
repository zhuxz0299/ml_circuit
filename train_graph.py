import torch
import torch.nn as nn
import torch.nn.functional as F
from torch_geometric.nn import GCNConv

class GCN(nn.Module):
    def __init__(self, input_dim, hidden_dim, output_dim):
        super(GCN, self).__init__()
        self.conv1 = GCNConv(input_dim, hidden_dim)
        self.conv2 = GCNConv(hidden_dim, output_dim)
        self.fc = nn.Linear(output_dim, 1)

    def forward(self, x, edge_index):
        x = self.conv1(x, edge_index)
        x = F.relu(x)
        x = self.conv2(x, edge_index)
        x = F.relu(x)
        x = torch.mean(x, dim=0)  # Graph Pooling
        x = self.fc(x)
        return x

# 构建图神经网络模型
input_dim = 710
hidden_dim = 128
output_dim = 64
model = GCN(input_dim, hidden_dim, output_dim)

# 定义损失函数和优化器
criterion = nn.MSELoss()
optimizer = torch.optim.Adam(model.parameters(), lr=0.01)

# 准备输入数据
node_features = torch.randn(710, input_dim)  # 节点特征
edge_index = torch.tensor([[0, 1, 2], [1, 2, 0]], dtype=torch.long)  # 边的索引
target = torch.tensor([7], dtype=torch.float32)  # 目标数字

# 训练模型
epochs = 100
for epoch in range(epochs):
    optimizer.zero_grad()
    output = model(node_features, edge_index)
    loss = criterion(output, target)
    loss.backward()
    optimizer.step()

    if (epoch+1) % 10 == 0:
        print(f"Epoch {epoch+1}/{epochs}, Loss: {loss.item()}")

# 进行预测
output = model(node_features, edge_index)
print("预测结果:", output.item())