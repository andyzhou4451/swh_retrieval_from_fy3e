import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader
import numpy as np
import zipfile
import torch.optim as optim
from c2net.context import prepare, upload_output
import random
import os

# 初始化上下文
c2net_context = prepare()
device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

# 设置随机种子
torch.manual_seed(0)
np.random.seed(0)
random.seed(0)

class VisionTransformerSharedLayers(nn.Module):
    def __init__(self, input_dim, seq_len, d_model, nhead, num_encoder_layers, dim_feedforward, patch_size=4):
        super(VisionTransformerSharedLayers, self).__init__()
        self.patch_size = patch_size
        self.num_patches = (input_dim // patch_size) * (seq_len // patch_size)
        self.embedding = nn.Linear(patch_size * patch_size, d_model)
        self.positional_encoding = nn.Parameter(torch.zeros(1, self.num_patches, d_model))
        
        encoder_layer = nn.TransformerEncoderLayer(d_model=d_model, nhead=nhead, dim_feedforward=dim_feedforward)
        self.transformer_encoder = nn.TransformerEncoder(encoder_layer, num_layers=num_encoder_layers)

    def forward(self, x):
        batch_size = x.size(0)
        # 划分图像为图块
        x = x.unfold(2, self.patch_size, self.patch_size).unfold(3, self.patch_size, self.patch_size)
        x = x.contiguous().view(batch_size, -1, self.patch_size * self.patch_size)  # (batch_size, num_patches, patch_size*patch_size)
        
        x = self.embedding(x)  # (batch_size, num_patches, d_model)
        x += self.positional_encoding[:, :x.size(1), :]  # 加入位置编码
        
        x = x.permute(1, 0, 2)  # (num_patches, batch_size, d_model)
        x = self.transformer_encoder(x)
        x = x.permute(1, 0, 2).contiguous().view(batch_size, -1)  # (batch_size, num_patches * d_model)
        return x

class MultiTaskModel(nn.Module):
    def __init__(self, input_dim, seq_len, d_model, nhead, num_encoder_layers, dim_feedforward, patch_size):
        super(MultiTaskModel, self).__init__()
        self.transformer_shared_layers = VisionTransformerSharedLayers(input_dim, seq_len, d_model, nhead, num_encoder_layers, dim_feedforward, patch_size)
        self.linear_shared_layer = nn.Sequential(
            nn.Linear((input_dim // patch_size) * (seq_len // patch_size) * d_model + 31, 1000),  # 修改此处的输入维度
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
        data2 = self.transformer_shared_layers(data2)
        data1 = data1.view(data1.size(0), -1)  # 确保 data1 是 (batch_size, 31)
        x = torch.cat((data1, data2), dim=1)
        x = self.linear_shared_layer(x)
        swh_output = self.wind_layer(x)
        return swh_output

# 模型参数
input_dim = 20  # 输入特征数
seq_len = 9     # 序列长度
d_model = 64    # Transformer编码器维度
nhead = 8       # 多头注意力头数
num_encoder_layers = 3  # Transformer编码器层数
dim_feedforward = 256   # 前馈神经网络的维度
patch_size = 4  # 图块大小

# 初始化模型
model = MultiTaskModel(input_dim, seq_len, d_model, nhead, num_encoder_layers, dim_feedforward, patch_size).to(device)

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
    L2_reg = sum(torch.norm(param, 2) for param in model.parameters())
    total_loss = loss + lambda_value * L2_reg
    return total_loss

# 数据集路径
swh_dataset_path = os.path.join(c2net_context.dataset_path, "processed_data_latlon.zip")
npz_file_name = 'processed_data_latlon.npz'

# 解压并加载数据
with zipfile.ZipFile(swh_dataset_path, 'r') as zip_ref:
    with zip_ref.open(npz_file_name) as npz_file:
        data = np.load(npz_file)
        scaled_data1 = data["scaled_data1"]
        scaled_data2 = data["scaled_data2"]
        swh = data["swh"]
        lon = data['lon']
        lat = data['lat']

# 确保 data1 的维度是 (n, 31) 和 data2 的维度是 (n, 9, 20)
scaled_data2 = scaled_data2.reshape(-1, 1, 9, 20)
scaled_data1 = scaled_data1.reshape(-1, 31)

# 数据集划分
np.random.seed(42)
indices = np.arange(scaled_data1.shape[0])
np.random.shuffle(indices)
scaled_data1 = scaled_data1[indices]
scaled_data2 = scaled_data2[indices]
swh = swh[indices]
lat = lat[indices]
lon = lon[indices]

train_por = 0.7
validation_por = 0.2
test_por = 0.1

train_size = int(len(scaled_data1) * train_por)
valid_size = int(len(scaled_data1) * validation_por)
test_size = len(scaled_data1) - train_size - valid_size

train_x1, train_x2, train_swh = scaled_data1[:train_size], scaled_data2[:train_size], swh[:train_size]
valid_x1, valid_x2, valid_swh = scaled_data1[train_size:train_size + valid_size], scaled_data2[train_size:train_size + valid_size], swh[train_size:train_size + valid_size]
test_x1, test_x2, test_swh, test_lon, test_lat = scaled_data1[train_size + valid_size:], scaled_data2[train_size + valid_size:], swh[train_size + valid_size:], lon[train_size + valid_size:], lat[train_size + valid_size:]

trainset = MyDataset(train_x1, train_x2, train_swh)
validset = MyDataset(valid_x1, valid_x2, valid_swh)
testset = MyDataset(test_x1, test_x2, test_swh)

train_loader = DataLoader(trainset, batch_size=512, shuffle=True)
valid_loader = DataLoader(validset, batch_size=512, shuffle=False)
test_loader = DataLoader(testset, batch_size=512, shuffle=False)

criterion = nn.MSELoss()
optimizer = optim.Adam(model.parameters(), lr=0.0005)

def train_and_evaluate_model(train_loader, valid_loader, test_loader, model, criterion, optimizer, num_epochs=500):
    train_losses, valid_losses, test_losses = [], [], []
    output_path = "./output"
    os.makedirs(output_path, exist_ok=True)

    for epoch in range(num_epochs):
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

        model.eval()
        valid_loss = 0.0
        with torch.no_grad():
            for data1, data2, labels in valid_loader:
                outputs = model(data1, data2).squeeze()
                valid_loss += criterion(outputs, labels).item()
        valid_loss /= len(valid_loader)
        valid_losses.append(valid_loss)

        test_loss = 0.0
        with torch.no_grad():
            for data1, data2, labels in test_loader:
                outputs = model(data1, data2).squeeze()
                test_loss += criterion(outputs, labels).item()
        test_loss /= len(test_loader)
        test_losses.append(test_loss)

        print(f'Epoch {epoch+1}/{num_epochs}, Train Loss: {train_loss:.4f}, Valid Loss: {valid_loss:.4f}, Test Loss: {test_loss:.4f}')

        state = {
            'epoch': epoch,
            'model_state': model.state_dict(),
            'optimizer_state': optimizer.state_dict(),
        }
        torch.save(state, f'{output_path}/model_epoch{epoch}.pkl')

    # Save final model
    final_model_path = os.path.join(c2net_context.output_path, 'final_model.pth')
    torch.save(model.state_dict(), final_model_path)
    print(f"Final model saved at epoch {num_epochs}")

    result_file = os.path.join(c2net_context.output_path, "training_results.txt")
    with open(result_file, 'w') as f:
        f.write("Training complete.\n")
        f.write("Epoch\tTrain Loss\tValid Loss\tTest Loss\n")
        for i, (t, v, te) in enumerate(zip(train_losses, valid_losses, test_losses), 1):
            f.write(f"{i}\t{t:.4f}\t{v:.4f}\t{te:.4f}\n")

    print('Training completed.')
    print("Results saved to", result_file)

    return train_losses, valid_losses, test_losses

train_losses, valid_losses, test_losses = train_and_evaluate_model(train_loader, valid_loader, test_loader, model, criterion, optimizer)

test_predictions = []
model.eval()
with torch.no_grad():
    for data1, data2, _ in test_loader:
        predictions = model(data1, data2).squeeze()
        test_predictions.extend(predictions.cpu().numpy())

np.savez(os.path.join(c2net_context.output_path, 'final_results_swh.npz'),
         train_losses=train_losses,
         valid_losses=valid_losses,
         test_losses=test_losses,
         test_predictions=test_predictions,
         test_swh=test_swh,
         test_data1=test_x1,
         test_data2=test_x2,
         lon=test_lon,
         lat=test_lat)

upload_output()

print("done")