# -*- coding: utf-8 -*-
import pandas as pd
from sklearn import metrics
from sklearn.model_selection import train_test_split
import xgboost as xgb
import matplotlib.pyplot as plt
import seaborn as sns
all_data_set_path = r'自跟踪数据俯仰.xlsx'
all_data_set = pd.read_excel(all_data_set_path)
print(all_data_set.head())
print(all_data_set.info())
print(all_data_set.isnull().sum())
# data = all_data_set.loc[:, ['方位座架命令角', '俯仰DA', '方位误差电压2']]
# target = all_data_set.loc[:, ['方位座架角']]
# print(data)
# print(target)
#
# train_x, test_x, train_y, test_y = train_test_split(data, target, test_size=0.2, random_state=7)
# # xgboost模型初始化设置
# dtrain =xgb.DMatrix(train_x, label=train_y)
# dtest = xgb.DMatrix(test_x)
# watchlist = [(dtrain, 'train')]
#
# # booster:
# params = {
#     'booster':'gbtree',
#     'objective': 'binary:logistic',
#     'eval_metric': 'auc',
#     'max_depth':5,
#     'lambda':10,
#     'subsample':0.75,
#     'colsample_bytree':0.75,
#     'min_child_weight':2,
#     'eta': 0.025,
#     'seed':0,
#     'nthread':8,
#     'gamma':0.15,
#     'learning_rate' : 0.01
# }
# bst = xgb.train(params, dtrain, num_boost_round=50, evals=watchlist)
# ypred = bst.predict(dtest)
plt.rcParams['font.sans-serif']=['SimHei']
plt.rcParams['axes.unicode_minus'] = False
plt.style.use('fivethirtyeight')

# 特征热力图 相关性分析
list_columns = all_data_set.columns
plt.figure(figsize=(15, 10))
sns.heatmap(all_data_set[list_columns].corr(), annot=True, fmt=".2f")
plt.show()
# 对特征重要性进行排序
corr_1 = all_data_set.corr()
print(corr_1["方位座架角"].sort_values(ascending=False))







