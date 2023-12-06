import torch
import torch.nn as nn
from flask import Flask, request, jsonify
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

app = Flask(__name__)

# 加载模型
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

# 特征提取函数
@app.route('/predict_csv', methods=['POST'])
def predict_csv():
    # 检查请求中是否包含文件
    if 'file' not in request.files:
        return jsonify({'error': 'No file part'})

    file = request.files['file']

    # 检查文件名是否为空
    if file.filename == '':
        return jsonify({'error': 'No selected file'})

    # 读取 CSV 文件
    try:
        df = pd.read_excel(file)
    except pd.errors.EmptyDataError:
        return jsonify({'error': 'Empty file'})

    data = np.array(df)
    data_x = np.array(data)[:, 1:]
    data_y = np.array(data)[:, 0]

    # 调用GPU
    device = torch.device('cuda') if torch.cuda.is_available() else torch.device('cpu')
    train_x = torch.from_numpy(data_x).to(torch.float).to(device)
    # 模型预测
    pre = model(train_x)
    pre = pre.cpu().data.numpy()

    # 返回预测结果
    # return jsonify({'predictions': pre.tolist()})
    return jsonify(pre.tolist())


if __name__ == '__main__':
    app.run(port=5000)

