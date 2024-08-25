import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader
import numpy as np
# import matplotlib.pyplot as plt
import os

# 模拟c2net context设置，真实使用时请替换为实际的context
class FakeContext:
    def __init__(self):
        self.device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

c2net_context = FakeContext()
device = c2net_context.device

# 定义模型
class MultiTaskModel(nn.Module):
    def __init__(self):
        super(MultiTaskModel, self).__init__()
        self.layer1 = nn.Linear(10, 50)
        self.layer2 = nn.Linear(50, 1)
    
    def forward(self, x):
        x = torch.relu(self.layer1(x))
        x = self.layer2(x)
        return x

# 定义数据集
class MyDataset(Dataset):
    def __init__(self, size=1000):
        self.data = torch.randn(size, 10)
        self.labels = torch.randn(size, 1)

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        return self.data[idx], self.labels[idx]

# 创建数据集和数据加载器
train_dataset = MyDataset(size=800)
valid_dataset = MyDataset(size=100)
test_dataset = MyDataset(size=100)

train_loader = DataLoader(train_dataset, batch_size=64, shuffle=True)
valid_loader = DataLoader(valid_dataset, batch_size=64, shuffle=False)
test_loader = DataLoader(test_dataset, batch_size=64, shuffle=False)

# 训练和评估模型
def train_and_evaluate_model(train_loader, valid_loader, test_loader, model, criterion, optimizer, num_epochs=50):
    model.to(device)
    train_losses, valid_losses, test_losses = [], [], []
    best_valid_loss = float('inf')
    best_model_state = None

    for epoch in range(num_epochs):
        # 训练阶段
        model.train()
        total_loss = 0
        for data, labels in train_loader:
            data, labels = data.to(device), labels.to(device)
            optimizer.zero_grad()
            outputs = model(data)
            loss = criterion(outputs, labels)
            loss.backward()
            optimizer.step()
            total_loss += loss.item()
        avg_train_loss = total_loss / len(train_loader)
        train_losses.append(avg_train_loss)

        # 验证阶段
        model.eval()
        total_loss = 0
        with torch.no_grad():
            for data, labels in valid_loader:
                data, labels = data.to(device), labels.to(device)
                outputs = model(data)
                loss = criterion(outputs, labels)
                total_loss += loss.item()
        avg_valid_loss = total_loss / len(valid_loader)
        valid_losses.append(avg_valid_loss)
        
        # 测试阶段
        total_loss = 0
        with torch.no_grad():
            for data, labels in test_loader:
                data, labels = data.to(device), labels.to(device)
                outputs = model(data)
                loss = criterion(outputs, labels)
                total_loss += loss.item()
        avg_test_loss = total_loss / len(test_loader)
        test_losses.append(avg_test_loss)

        # 更新最佳模型（如果有）
        if avg_valid_loss < best_valid_loss:
            best_valid_loss = avg_valid_loss
            best_model_state = model.state_dict()

        print(f'Epoch {epoch + 1}: Train Loss: {avg_train_loss}, Valid Loss: {avg_valid_loss}, Test Loss: {avg_test_loss}')

    # 保存最佳模型
    if best_model_state:
        torch.save(best_model_state, 'best_model.pth')
        print(f'Best model saved with validation loss: {best_valid_loss}')

    # 绘制损失曲线
    # plt.figure(figsize=(10, 5))
    # plt.plot(train_losses, label='Train Loss')
    # plt.plot(valid_losses, label='Validation Loss')
    # plt.plot(test_losses, label='Test Loss')
    # plt.xlabel('Epochs')
    # plt.ylabel('Loss')
    # plt.title('Loss Curves')
    # plt.legend()
    # plt.savefig('loss_curves.png')
    # plt.show()

    return train_losses, valid_losses, test_losses

# 设置损失函数和优化器
model = MultiTaskModel()
criterion = nn.MSELoss()
optimizer = optim.Adam(model.parameters(), lr=0.001)

# 运行训练和评估
train_losses, valid_losses, test_losses = train_and_evaluate_model(train_loader, valid_loader, test_loader, model, criterion, optimizer)
