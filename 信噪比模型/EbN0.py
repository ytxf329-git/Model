"""
读取后处理的天线方向图，在原始数据范围内进行3阶样条差值，最后根据插值函数f(x)得出实际运行过程中的增益
"""
import pandas as pd
from pylab import *
import numpy as np
from scipy.interpolate import interp1d
import ephem
from datetime import datetime, timedelta
import math

"""
计算过境时卫星与地面站夹角&卫星与地面站之间距离，前者用于计算星上天线增益，后者用于计算空衰
"""
def ebn0(line1, line2, sate_power, sate_freq, Data_Rate, station_GT, observer_lat, observer_lon, observer_elev, starttime, endtime):
    global EbN0
    # 方向图加载及差值处理
    data = pd.read_csv(r'D:\work\tkinterGUI\tkinterGUI\V3.0\卫星方向图.csv')
    x = data.loc[:, 'theta']
    y = data.loc[:, 'gain']
    x_min = min(data.loc[:, 'theta'])
    x_max = max(data.loc[:, 'theta'])
    F = interp1d(x, y, kind='cubic')
    x_inter = np.linspace(x_min, x_max, 300)
    y_inter = F(x_inter)
    # 其余参数
    # 地球半径（单位：千米）
    earth_radius = 6371e3
    # 起跟角
    ele_start = 5
    ele_start = radians(ele_start)
    # 设置编码增益及译码损耗、误码率、指向误差损耗
    direction_loss = 0.5
    encode_gain = 1
    decode_loss = 0.5
    BER = 1e-6
    K = -228.6

    # 读取TLE数据,创建卫星对象
    satellite = ephem.readtle('SATE', line1, line2)
    # 设置地面站及参数
    observer = ephem.Observer()
    observer.lat = observer_lat
    observer.lon = observer_lon
    observer.elev = observer_elev
    station_GT = station_GT

    # 星上参数
    sate_power = sate_power
    sate_freq = sate_freq
    Data_Rate = Data_Rate
    # 计算卫星过境时的方位和俯仰角

    start_time = starttime # 替换为实际的开始时间
    end_time = endtime  # 替换为实际的结束时间

    t = start_time
    Az = []
    El = []
    Sate_Height = []
    Distance = []
    Theta = []
    Sate_EIRP = []
    EbN0 = []
    with open('data.txt', 'a', encoding='utf-8') as f:
        f.write(
            '时间' + "---" + '方位角' + "---" + '俯仰角' + "---" + '轨道高度' + "---" + '星地距离' + "---" + '天线方向角' + "---" + 'Eb/N0' + '\n')
    while t < end_time:
        observer.date = t
        satellite.compute(observer)
        az = satellite.az  # 方位角（以弧度为单位）
        el = satellite.alt  # 俯仰角（以弧度为单位）
        if el >= ele_start:
            Az.append(math.degrees(az))
            El.append(math.degrees(el))
            # 计算观测站与卫星之间的距离 km
            distance = satellite.range
            Distance.append(distance)
            # 计算轨道高度 km
            sate_Height = satellite.elevation
            Sate_Height.append(sate_Height)
            # 计算卫星与地面站夹角°
            theta = math.asin(earth_radius * math.sin(math.pi / 2 + el) / (earth_radius + satellite.elevation))
            Theta.append(math.degrees(theta))
            # 计算卫星天线增益 dBW
            sate_EIRP = F(theta) + sate_power
            Sate_EIRP.append(sate_EIRP)
            # 计算空间损耗
            link_loss = 20 * (math.log10(sate_freq) + math.log10(distance / 1000)) + 32.44
            # 计算接收C/N0
            CN0 = sate_EIRP + station_GT - link_loss - direction_loss - K
            # 计算接收的Eb/N0
            ebn0 = CN0 - 10 * math.log10(Data_Rate)
            EbN0.append(ebn0)
        t += timedelta(seconds=1)
    return EbN0







#
#
# fig = plt.figure(1)
#
# # 用户数据
# ax1 = plt.subplot(4, 2, 1)
# plt.plot(x, y, 'o')
# plt.legend(['data',  'nearest'], loc='best')
# plt.xlabel(u'theta', size=12)
# plt.ylabel(u'gain', size=12)
# # 样条插值
# ax2 = plt.subplot(4, 2, 2)
# plt.plot(x, y, 'o', x_inter, F(x_inter), '-')
# plt.legend(['data', 'cubic', 'nearest'], loc='best')
# plt.xlabel(u'theta', size=12)
# plt.ylabel(u'gain', size=12)
# # 方位角
# ax3 = plt.subplot(4, 2, 3)
# plt.plot(Az)
# plt.legend(['Az', 'nearest'], loc='best')
# plt.xlabel(u'T', size=12)
# plt.ylabel(u'Angle', size=12)
# # 俯仰角
# ax4 = plt.subplot(4, 2, 4)
# plt.plot(El)
# plt.legend(['El', 'nearest'], loc='best')
# plt.xlabel(u'T', size=12)
# plt.ylabel(u'Angle', size=12)
# # 轨道高度
# ax5 = plt.subplot(4, 2, 5)
# plt.plot(Sate_Height)
# plt.legend(['Sate_Height', 'nearest'], loc='best')
# plt.xlabel(u'T', size=12)
# plt.ylabel(u'Km', size=12)
# # 星地距离
# ax6 = plt.subplot(4, 2, 6)
# plt.plot(Distance)
# plt.legend(['Distance', 'nearest'], loc='best')
# plt.xlabel(u'T', size=12)
# plt.ylabel(u'Km', size=12)
# # 天线方向角
# ax7 = plt.subplot(4, 2, 7)
# plt.plot(Theta)
# plt.legend(['Theta', 'nearest'], loc='best')
# plt.xlabel(u'T', size=12)
# plt.ylabel(u'°', size=12)
# # Eb/N0
# ax8 = plt.subplot(4, 2, 8)
# plt.plot(EbN0)
# plt.legend(['Eb/N0', 'nearest'], loc='best')
# plt.xlabel(u'T', size=12)
# plt.ylabel(u'dB', size=12)
#
# plt.show()
