import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader
import numpy as np
import zipfile
import torch.optim as optim
from c2net.context import prepare
import random
import os
# 初始化上下文
c2net_context = prepare()
device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

torch.manual_seed(0)
np.random.seed(0)
random.seed(0)


class MultiTaskModel(nn.Module):
    def __init__(self):
        super(MultiTaskModel, self).__init__()
        self.conv_shared_layers = nn.Sequential(
            nn.Conv2d(1, 128, 5),
            nn.ReLU(),
            nn.Conv2d(128, 32, 4),
            nn.ReLU(),
            nn.Conv2d(32, 1, 2),
            nn.BatchNorm2d(1),
            nn.ReLU()
        )
        self.linear_shared_layer = nn.Sequential(
            # nn.Linear(, 1000),
            nn.Linear(31, 1000),
            nn.BatchNorm1d(1000),
            nn.ReLU(),
            nn.Dropout(p=0.3)
        )
        self.wind_layer = nn.Sequential(
            nn.Linear(1000, 2000),
            nn.BatchNorm1d(2000),
            nn.ReLU(),
            nn.Dropout(p=0.2),
            nn.Linear(2000, 1500),
            nn.BatchNorm1d(1500),
            nn.ReLU(),
            nn.Dropout(p=0.2),
            nn.Linear(1500, 500),
            nn.BatchNorm1d(500),
            nn.ReLU(),
            nn.Dropout(p=0.2),
            nn.Linear(500, 200),
            nn.BatchNorm1d(200),
            nn.ReLU(),
            nn.Dropout(p=0.2),
            nn.Linear(200, 100),
            nn.BatchNorm1d(100),
            nn.ReLU(),
            nn.Dropout(p=0.2),
            nn.Linear(100, 10),
            nn.BatchNorm1d(10),
            nn.ReLU(),
            nn.Dropout(p=0.2),
            nn.Linear(10, 1)
        )

    def forward(self, data1, data2):
        # data2 = self.conv_shared_layers(data2)
        # x = torch.flatten(data2, 1)
        # x = torch.cat((data1, data2), dim=1)
        x = self.linear_shared_layer(data1)
        swh_output = self.wind_layer(x)
        return swh_output

class MyDataset(Dataset):
    def __init__(self, data1, data2, labels):
        self.data1 = torch.tensor(data1, dtype=torch.float32).to(device)
        self.data2 = torch.tensor(data2, dtype=torch.float32).to(device)
        self.labels = torch.tensor(labels, dtype=torch.float32).to(device)

    def __len__(self):
        return len(self.labels)

    def __getitem__(self, idx):
        return self.data1[idx], self.data2[idx], self.labels[idx]

def calculate_loss(model, criterion, data1, data2, labels, lambda_value=0.01):
    outputs = model(data1, data2).squeeze()
    loss = criterion(outputs, labels)
    L2_reg = torch.tensor(0., device=device)
    for param in model.parameters():
        L2_reg += torch.norm(param, 2)
    total_loss = loss + lambda_value * L2_reg
    return total_loss

# 这里添加加载数据集的代码，创建 DataLoader 等步骤
swh_dataset_path = c2net_context.dataset_path + "/" + "processed_data.zip" 
npz_file_name = 'processed_data.npz'
with zipfile.ZipFile(swh_dataset_path, 'r') as zip_ref:
    with zip_ref.open(npz_file_name) as npz_file:
        data = np.load(npz_file)
        scaled_data1 = data["scaled_data1"]
        scaled_data2 = data["scaled_data2"]
        swh = data["swh"]

# 预处理数据
scaled_data2 = scaled_data2.reshape(-1, 1, 9, 20)
# indices = np.arange(scaled_data1.shape[0])

np.random.seed(42)  # 设置固定种子以保证每次打乱一致
indices = np.arange(scaled_data1.shape[0])
np.random.shuffle(indices)
scaled_data1 = scaled_data1[indices]
scaled_data2 = scaled_data2[indices]
swh = swh[indices]

# 更新训练集、验证集和测试集的比例
train_por = 0.7
validation_por = 0.2  # 更新为0.2
test_por = 0.1  # 新增测试集占10%

train_size = int(scaled_data1.shape[0] * train_por)
valid_size = int(scaled_data1.shape[0] * validation_por)
test_size = int(scaled_data1.shape[0] * test_por)

train_x1 = scaled_data1[:train_size]
train_x2 = scaled_data2[:train_size]
train_swh = swh[:train_size]

valid_x1 = scaled_data1[train_size:train_size + valid_size]
valid_x2 = scaled_data2[train_size:train_size + valid_size]
valid_swh = swh[train_size:train_size + valid_size]

test_x1 = scaled_data1[train_size + valid_size:]
test_x2 = scaled_data2[train_size + valid_size:]
test_swh = swh[train_size + valid_size:]

# 创建 DataLoader
trainset = MyDataset(train_x1, train_x2, train_swh)
train_loader = DataLoader(trainset, batch_size=512, shuffle=True, drop_last=False, pin_memory=False, num_workers=0)
validset = MyDataset(valid_x1, valid_x2, valid_swh)
valid_loader = DataLoader(validset, batch_size=512, shuffle=True, drop_last=False, pin_memory=False, num_workers=0)
testset = MyDataset(test_x1, test_x2, test_swh)
test_loader = DataLoader(testset, batch_size=512, shuffle=False, drop_last=False, pin_memory=False, num_workers=0)

model = MultiTaskModel().to(device)
criterion = torch.nn.MSELoss()
optimizer = optim.Adam(model.parameters(), lr=0.0005)

def train_and_evaluate_model(train_loader, valid_loader, test_loader, model, criterion, optimizer, num_epochs=500):
    train_losses, valid_losses, test_losses = [], [], []
    best_valid_loss = float('inf')
    output_path = "./output"  # 根据您的环境设置适当的路径
    os.makedirs(output_path, exist_ok=True)

    for epoch in range(num_epochs):
        # 训练阶段
        model.train()
        running_loss = 0.0
        for data1, data2, labels in train_loader:
            optimizer.zero_grad()
            outputs = model(data1, data2).squeeze()
            loss = criterion(outputs, labels)
            loss.backward()
            optimizer.step()
            running_loss += loss.item()
        train_loss = running_loss / len(train_loader)
        train_losses.append(train_loss)

        # 验证阶段
        model.eval()
        valid_loss = 0.0
        with torch.no_grad():
            for data1, data2, labels in valid_loader:
                outputs = model(data1, data2).squeeze()
                valid_loss += criterion(outputs, labels).item()
        valid_loss = valid_loss / len(valid_loader)
        valid_losses.append(valid_loss)

        # 测试阶段
        test_loss = 0.0
        with torch.no_grad():
            for data1, data2, labels in test_loader:
                outputs = model(data1, data2).squeeze()
                test_loss += criterion(outputs, labels).item()
        test_loss = test_loss / len(test_loader)
        test_losses.append(test_loss)

        # 打印训练信息
        print(f'Epoch {epoch+1}/{num_epochs}, Train Loss: {train_loss:.4f}, Valid Loss: {valid_loss:.4f}, Test Loss: {test_loss:.4f}')

        # 保存最佳模型
        if valid_loss < best_valid_loss:
            best_valid_loss = valid_loss
            torch.save(model.state_dict(), os.path.join(output_path, 'best_model.pth'))
            print(f"Best model saved at epoch {epoch+1} with validation loss {best_valid_loss:.4f}")


    # 保存训练过程到文件
    with open(os.path.join(output_path, 'training_results.txt'), 'w') as f:
        f.write("Training complete. Best Validation Loss: {:.4f}\n".format(best_valid_loss))
        for i, (t, v, te) in enumerate(zip(train_losses, valid_losses, test_losses), 1):
            f.write(f"Epoch {i}: Train Loss: {t:.4f}, Valid Loss: {v:.4f}, Test Loss: {te:.4f}\n")

    return train_losses, valid_losses, test_losses, best_valid_loss
# 运行训练和评估
train_losses, valid_losses, test_losses, best_valid_loss = train_and_evaluate_model(train_loader, valid_loader, test_loader, model, criterion, optimizer)

print('Training complete. Best Validation Loss:', best_valid_loss)
print('Test Losses:', test_losses)


# 将模型保存到c2net_context.output_path
state = {'model':model.state_dict(), 'optimizer':optimizer.state_dict(), 'epoch':epoch}
torch.save(state, '{}/swh_cnn_epoch{}.pkl'.format(c2net_context.output_path, epoch))

result_file = os.path.join(c2net_context.output_path, "swh_cnn_training_results.txt")
with open(result_file, 'w') as f:
    f.write("训练完成。最佳验证损失: {:.4f}\n".format(best_valid_loss))
    f.write("每轮训练损失: {}\n".format(train_losses))
    f.write("每轮验证损失: {}\n".format(valid_losses))

print('训练完成。最佳验证损失:', best_valid_loss)
print("结果已保存到", result_file)



print("done")
