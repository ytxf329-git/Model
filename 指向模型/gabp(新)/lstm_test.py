import torch
import torch.nn as nn
import torch.optim as optim
import numpy as np
import pandas as pd
from torch.utils.data import TensorDataset, DataLoader

# 数据准备
df = pd.read_excel('Az_data.xlsx')
data = np.array(df)
np.random.shuffle(data)
data_x = np.array(data)[:, 1:]
data_y = np.array(data)[:, 0]
N = data_x.shape[0]

train_x = data_x[:int(N * 0.8), :]
train_y = data_y[:int(N * 0.8)]
test_x = data_x[int(N * 0.8):, :]
test_y = data_y[int(N * 0.8):]

# device = torch.device('cuda') if torch.cuda.is_available() else torch.device('cpu')
# X = torch.from_numpy(train_x).to(torch.float).to(device)
# Y = torch.from_numpy(train_y.reshape(-1, 1)).to(torch.float).to(device)
X = torch.from_numpy(train_x).to(torch.float)
Y = torch.from_numpy(train_y.reshape(-1, 1)).to(torch.float)


# 标准化数据
mean = X.mean()
std = X.std()
X = (X - mean) / std

class LSTMModel(nn.Module):
    def __init__(self, input_dim, hidden_dim, num_layers, output_dim):
        super(LSTMModel, self).__init__()
        self.input_dim = input_dim
        self.hidden_dim = hidden_dim
        self.num_layers = num_layers
        self.lstm = nn.LSTM(input_dim, hidden_dim, num_layers, batch_first=True)
        self.fc = nn.Linear(hidden_dim, output_dim)

    def forward(self, x):
        # print(x.shape)
        h0 = torch.zeros(self.num_layers, x.size(0), self.hidden_dim).to(x.device)
        c0 = torch.zeros(self.num_layers, x.size(0), self.hidden_dim).to(x.device)
        # print(x.shape)
        # print(h0.shape, c0.shape)
        out, _ = self.lstm(x.view(-1,1,self.input_dim), (h0, c0))
        # out, _ = self.lstm(x, (h0, c0))
        out = self.fc(out[:, -1, :])  # 取序列的最后一个时间步的输出
        return out

input_dim = 3  # 输入特征的维度，这里是3
hidden_dim = 50  # LSTM隐藏层大小
num_layers = 2  # LSTM层数
output_dim = 1  # 输出维度，这里是1
model = LSTMModel(input_dim, hidden_dim, num_layers, output_dim)

criterion = nn.MSELoss()
optimizer = optim.Adam(model.parameters(), lr=1e-2)


# 创建数据集和数据加载器
train_dataset = TensorDataset(X, Y)
batch_size = 32  # 选择适当的批次大小
train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
num_epochs = 1000
for epoch in range(num_epochs):
    for batch_X, batch_Y in train_loader:
        outputs = model(batch_X)
        optimizer.zero_grad()
        loss = criterion(outputs, batch_Y)
        loss.backward()
        optimizer.step()

    if (epoch + 1) % 10 == 0:
        print(f'Epoch [{epoch + 1}/{num_epochs}], Loss: {loss.item():.4f}')


