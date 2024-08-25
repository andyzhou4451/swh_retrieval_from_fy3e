import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader
import numpy as np
import zipfile
import torch.optim as optim
from c2net.context import prepare
from c2net.context import upload_output
import random
import os

# 初始化上下文
c2net_context = prepare()
device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

torch.manual_seed(2)
np.random.seed(2)
random.seed(2)

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
            nn.Linear(43, 1000),
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

    # def forward(self, data1, data2):
    #     data2 = self.conv_shared_layers(data2)
    #     x = torch.flatten(data2, 1)
    #     x = self.linear_shared_layer(x)
    #     swh_output = self.wind_layer(x)
    #     return swh_output
    def forward(self, data1, data2):
        data2 = self.conv_shared_layers(data2)
        data2 = torch.flatten(data2, 1)
        x = torch.cat((data1, data2), dim=1)
        x = self.linear_shared_layer(x)
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

swh_dataset_path = c2net_context.dataset_path + "/" + "processed_data_latlontime.zip" 
npz_file_name = 'processed_data_latlontime.npz'
with zipfile.ZipFile(swh_dataset_path, 'r') as zip_ref:
    with zip_ref.open(npz_file_name) as npz_file:
        data = np.load(npz_file)
        scaled_data1 = data["scaled_data1"]
        scaled_data2 = data["scaled_data2"]
        swh = data["swh"]
        lon = data['lon']
        lat = data['lat']
        time = data['time']

scaled_data2 = scaled_data2.reshape(-1, 1, 9, 20)

np.random.seed(42)
indices = np.arange(scaled_data1.shape[0])
np.random.shuffle(indices)
scaled_data1 = scaled_data1[indices]
scaled_data2 = scaled_data2[indices]
swh = swh[indices]
lat = lat[indices]
lon = lon[indices]
time = time[indices]

train_por = 0.7
validation_por = 0.2
test_por = 0.1

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
test_lon = lon[train_size + valid_size:]
test_lat = lat[train_size + valid_size:]
test_time = time[train_size + valid_size:]


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
        valid_loss = valid_loss / len(valid_loader)
        valid_losses.append(valid_loss)

        test_loss = 0.0
        with torch.no_grad():
            for data1, data2, labels in test_loader:
                outputs = model(data1, data2).squeeze()
                test_loss += criterion(outputs, labels).item()
        test_loss = test_loss / len(test_loader)
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
        f.write(f"Training complete.\n")
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
         lon = test_lon,
         lat = test_lat,
         time = test_time)

# Create zip file
# zip_filename = os.path.join(c2net_context.output_path, "model_and_results.zip")
# with zipfile.ZipFile(zip_filename, 'w') as zipf:
#     zipf.write(final_model_path, os.path.basename(final_model_path))
#     zipf.write(os.path.join(c2net_context.output_path, 'final_results.npz'), 'final_results.npz')
#     zipf.write(result_file, 'training_results.txt')

upload_output()

print("done")
