from keras.datasets import boston_housing
from keras import models
from keras import layers

# (X_train, y_train), (X_test, y_test) = boston_housing.load_data()  # 加载数据

#导入标签数据集

import numpy as np
import pandas as pd
import os

from keras.utils import plot_model

target_csv = pd.read_csv(r"data\unrestricted_imei1_5_6_2019_20_9_17.csv", usecols=(0, 144))
# print("type of df", type(target_csv))
value = target_csv.values
# print("type of value: ", type(value))
# print(value)
# print("shape of value: ", value.shape)

#导入特征数据集
from scipy.io import loadmat

feature_mat = loadmat("data\HCP_pCorr_1003s_100ROI.mat")
ID_mat = loadmat("data\HCP_1003s_fMRI_ID.mat")

# 把csv里的标签对应添加到ID_mat字典里
temporary = np.zeros((1003,1))
i = 0
j = 0
flag = 0

while i < 1003:
    j = 0
    while j < 1206:
        if int(value[j][0]) == ID_mat['fMRI_ID'][0][i]:
            # print(value[j][1])
            # print(temporary[i][0])
            temporary[i][0] = value[j][1]
            break
        j += 1
    i += 1

feature_mat['target'] = temporary
print(feature_mat['target'])

predict_null = pd.isnull(feature_mat['target'])
k = 0
while k < 1003:
    if predict_null[k][0] == True:
        print(k)
    k += 1


## 数据建模
#2.1 数据集拆分
#导入数据集拆分工具
from sklearn.model_selection import train_test_split

# 将数据集的数值和分类目标赋值给x,y
feature_X, targer_Y = feature_mat['CorrVec'], feature_mat['target']
# 删除找到的空值
feature_X = np.delete(feature_X, [45, 523, 768], axis=0)
targer_Y = np.delete(targer_Y, [45, 523, 768], axis=0)

##数据预处理
# 数据标准化，使用preprocessing库的StandardScaler类对数据进行标准化
from sklearn.preprocessing import StandardScaler

# 标准化，返回值为标准化后的数据
feature_X = StandardScaler().fit_transform(feature_X)
# targer_Y = StandardScaler().fit_transform(targer_Y)
# 特征选择
from sklearn.feature_selection import SelectKBest
from scipy.stats import pearsonr
from sklearn.feature_selection import f_regression
feature_X = np.squeeze(feature_X)
targer_Y = np.squeeze(targer_Y)
## use Pearson
# sele_X = SelectKBest(lambda X,Y: np.array(list(map(lambda x: pearsonr(x, Y)[0], X.T))).T, k = 300).fit_transform(feature_X, targer_Y)
# 使用 f_regression
fregre_sele = SelectKBest(f_regression, k = 300)
sele_X = fregre_sele.fit_transform(feature_X, targer_Y)
# 使用 mutal_info_regression 互信息
# from minepy import MINE
# def mic(x, y):
#     m = MINE()
#     m.compute_score(x, y)
#     return (m.mic(), 0.5)
# sele_X = SelectKBest(lambda X, Y: np.array(list(map(lambda x: mic(x, Y)[0], X.T))).T, k = 300).fit_transform(feature_X, targer_Y)

# 拆分数据集---x,y都要拆分，rain_test_split(x,y,random_state=0),random_state=0使得每次生成的伪随机数不同
X_train, x_test, Y_train, y_test = train_test_split(sele_X, targer_Y, random_state=0)
print('x_train_shape:{}'.format(X_train.shape))
print('x_test_shape:{}'.format(x_test.shape))
print('y_train_shape:{}'.format(Y_train.shape))
print('y_test_shape:{}'.format(y_test.shape))
print('\n')

# # 对数据进行标准化预处理，方便神经网络更好的学习
# mean = X_train.mean(axis=0)
# X_train -= mean
# std = X_train.std(axis=0)
# X_train /= std
# X_test -= mean
# X_test /= std

from keras.layers import Dense, Dropout
# 构建神经网络模型
seed = 7
np.random.seed(seed)

def build_model():
    # 这里使用Sequential模型
    model = models.Sequential()
    # 进行层的搭建，注意第二层往后没有输入形状(input_shape)，它可以自动推导出输入的形状等于上一层输出的形状
    model.add(layers.Dense(64, activation='relu', input_shape=(X_train.shape[1],)))
    model.add(Dropout(0.2))
    model.add(layers.Dense(64, activation='relu'))
    model.add(Dropout(0.2))
    model.add(layers.Dense(64, activation='relu'))
    model.add(Dropout(0.2))
    model.add(layers.Dense(1))
    # 编译网络
    model.compile(optimizer='adam', loss='mse', metrics=['mae'])
    plot_model(model, to_file='./model_SimpleDNN.png', show_shapes=True)
    return model


num_epochs = 100
model = build_model()
model.fit(X_train, Y_train, epochs=num_epochs, batch_size=16, verbose=1)
predicts = model.predict(x_test)

# 评估模型分数
import numpy as np
from sklearn import metrics
MSE = metrics.mean_squared_error(y_test, predicts)
RMSE = np.sqrt(metrics.mean_squared_error(y_test, predicts))
R2 = metrics.r2_score(y_test, predicts)
# R = np.sqrt(metrics.r2_score(y_test, predicts))
# data = pd.DataFrame(y_test)
# data.insert(1, predicts)
# print(data)
# Corr = data['0'].corr(data['1'])
# print("Corr:", Corr)
print('MSE:', MSE)
print('RMSE:', RMSE)
print("R^2:", R2)
# print("R:", R)

data = pd.DataFrame(y_test)
data['2'] = predicts
# data.rename(columns={'0':'y_test', '2':'predicts'})
data.columns = ['y_test', 'predicts']
# print(data)
Corr = data['y_test'].corr(data['predicts'])
print("Corr:", Corr)

# 画画图
import matplotlib.pyplot as plt
from scipy import optimize


def f_1(x, A, B):
    return A * x + B

predicts = np.squeeze(predicts)
A1, B1 = optimize.curve_fit(f_1, y_test, predicts)[0]
nihe_x = y_test
nihe_y = A1 * nihe_x + B1
plt.plot(nihe_x, nihe_y, color='b', linewidth=2.0, label='Predicted Label')
plt.title("SimpleDNN")
plt.scatter(y_test, predicts)
plt.xlabel("True Values")
plt.ylabel("Predictions")
plt.plot(y_test, y_test, color='r', linewidth=1.0, linestyle='-.', label='y = x')
plt.text(2, 23, 'Corr=' + str(round(Corr, 3)))
# plt.legend()
ax = plt.gca()
ax.set_aspect('equal')
plt.show()