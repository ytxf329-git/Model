import torch
import torch.nn as nn
import EbN0
from flask import Flask, request, jsonify
from datetime import datetime
import pandas as pd
import numpy as np

app = Flask(__name__)
"-------------指向模型加载-----------------"
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
"-------------信噪比模型参数定义-----------------"
# 定义用到的键值
filepath_key = 'filepath'
line1_key = 'TLEline1'
line2_key = 'TLEline2'
sate_power_key = 'satepower'
sate_freq_key = 'satefreq'
Data_Rate_key = 'datarate'
station_GT_key = 'station_GT'
observer_lat_key = 'observer_lat'
observer_lon_key = 'observer_lon'
observer_elev_key = 'observer_elev'
starttime_key = 'starttime'
endtime_key = 'endtime'
date_format = "%Y, %m, %d, %H, %M, %S"

"-------------信噪比模型计算----------------------"
# 定义一个POST请求的路由
@app.route('/post_example', methods=['POST'])
def post_example():
    # 获取POST请求的数据
    data = request.get_json()

    # 假设请求中有一个名为 "message" 的字段
    if 'message' in data:
        message_array = data['message']
        response = {'status': 'success', 'message_received': message_array}
        for message in message_array:
            if filepath_key in message:
                filepath = message[filepath_key]
            if line1_key in message:
                line1 = message[line1_key]
            if line2_key in message:
                line2 = message[line2_key]
            if sate_power_key in message:
                sate_power = message[sate_power_key]
            if sate_freq_key in message:
                sate_freq = message[sate_freq_key]
            if Data_Rate_key in message:
                Data_Rate = message[Data_Rate_key]
            if station_GT_key in message:
                station_GT = message[station_GT_key]
            if observer_lat_key in message:
                observer_lat = message[observer_lat_key]
            if observer_lon_key in message:
                observer_lon = message[observer_lon_key]
            if observer_elev_key in message:
                observer_elev = message[observer_elev_key]
            if starttime_key in message:
                start_time = message[starttime_key]
                start_time = datetime.strptime(start_time, date_format)
            if endtime_key in message:
                end_time = message[endtime_key]
                end_time = datetime.strptime(end_time, date_format)
        ebn0 = EbN0.ebn0(filepath, line1, line2, sate_power,sate_freq, Data_Rate, station_GT, observer_lat, observer_lon, observer_elev, start_time, end_time)
    else:
        response = {'status': 'error', 'message': 'Missing "message" field'}
    return jsonify(ebn0)
    return jsonify(response)

"-------------指向模型计算----------------------"
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