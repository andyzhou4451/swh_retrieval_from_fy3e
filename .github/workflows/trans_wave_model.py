import ssl
import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader
import numpy as np
import zipfile
import torch.optim as optim
from c2net.context import prepare, upload_output
import random
import os
import timm

# 初始化上下文
c2net_context = prepare()
device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

# 设置随机种子
torch.manual_seed(0)
np.random.seed(0)
random.seed(0)

class SwinTransformerSharedLayers(nn.Module):
    def __init__(self, img_size=224, in_chans=1, embed_dim=96):
        super(SwinTransformerSharedLayers, self).__init__()
        self.swin_transformer = timm.create_model('swin_small_patch4_window7_224', pretrained=False, in_chans=in_chans)
        self.swin_transformer.head = nn.Identity()  # 去掉分类头

        # 解压并加载本地预训练权重
        swin_path = os.path.join(c2net_context.dataset_path, "swin_small_patch4_window7_224.zip")
        swin_file_name = 'swin_small_patch4_window7_224.pth'
        with zipfile.ZipFile(swin_path, 'r') as zip_ref:
            zip_ref.extract(swin_file_name, './')

        pretrained_weights_path = os.path.join('./', swin_file_name)
        if os.path.exists(pretrained_weights_path):
            state_dict = torch.load(pretrained_weights_path, map_location=device)
            # 处理嵌套的键
            if 'model' in state_dict:
                state_dict = state_dict['model']
            # 过滤掉不匹配的键
            model_state_dict = self.swin_transformer.state_dict()
            filtered_state_dict = {k: v for k, v in state_dict.items() if k in model_state_dict and v.size() == model_state_dict[k].size()}
            model_state_dict.update(filtered_state_dict)
            self.swin_transformer.load_state_dict(model_state_dict, strict=False)
        else:
            raise FileNotFoundError(f"预训练权重文件未找到：{pretrained_weights_path}")

    def forward(self, x):
        x = self.swin_transformer(x)
        return x

class MultiTaskModel(nn.Module):
    def __init__(self, img_size, embed_dim, input_dim):
        super(MultiTaskModel, self).__init__()
        self.transformer_shared_layers = SwinTransformerSharedLayers(img_size, in_chans=1, embed_dim=embed_dim)
        self.linear_shared_layer = nn.Sequential(
            nn.Linear(embed_dim + 31, 1000),  # 修改此处的输入维度
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
img_size = 224
embed_dim = 768  # swin_small_patch4_window7_224的嵌入维度
input_dim = 20  # 输入特征数

# 初始化模型
model = MultiTaskModel(img_size, embed_dim, input_dim).to(device)

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

# 确保 data1 的维度是 (n, 31) 和 data2 的维度是 (n, 1, 9, 20)
scaled_data2 = scaled_data2[:, np.newaxis, :, :]  # 添加一个新的维度，变为 (n, 1, 9, 20)
scaled_data2 = np.pad(scaled_data2, ((0, 0), (0, 0), (0, img_size - 9), (0, img_size - 20)), 'constant')  # 填充为 (n, 1, 224, 224)
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
