import torch
import torch.nn as nn
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

class new_model(nn.Module):

    def __init__(self) -> None:
        super().__init__()
        self.dimensionality_dense = nn.Sequential(
            nn.Linear(4, 10),
            nn.Tanh(),
            nn.Linear(10, 10),
            nn.Tanh(),
            nn.Linear(10, 10),
            nn.Tanh(),
            nn.Linear(10, 4),
            nn.Tanh())
        self.norm = nn.LayerNorm(4)
        self.dense = nn.Linear(4, 1)

    def forward(self, x):
        x = self.dimensionality_dense(x)
        x = self.norm(x)
        x = self.dense(x)
        return x


model = torch.load("model.pkl")
model.eval()
# 获取数据
df = pd.read_excel('方位角数据.xlsx')
data = np.array(df)
data_x = np.array(data)[:, 1:]
data_y = np.array(data)[:, 0]
# 调用GPU
device = torch.device('cuda') if torch.cuda.is_available() else torch.device('cpu')
train_x = torch.from_numpy(data_x).to(torch.float).to(device)
# 模型预测
pre = model(train_x)
pre = pre.cpu().data.numpy()
# 预测结果绘图
plt.figure()
plt.plot(data_y.flatten(), label='true')
plt.plot(pre, label='pred')
plt.legend()
plt.show()
