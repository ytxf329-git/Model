import EbN0
from flask import Flask, request, jsonify
from datetime import datetime

app = Flask(__name__)
# 定义用到的键值
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
        ebn0 = EbN0.ebn0(line1, line2, sate_power,sate_freq, Data_Rate, station_GT, observer_lat, observer_lon, observer_elev, start_time, end_time)
    else:
        response = {'status': 'error', 'message': 'Missing "message" field'}
    return jsonify(ebn0)
    return jsonify(response)

if __name__ == '__main__':
    app.run(debug=True)

