import matplotlib.pyplot as plt
from math import sqrt
from matplotlib import pyplot
import pandas as pd
import numpy as np
from sklearn.preprocessing import MinMaxScaler
from sklearn.metrics import mean_squared_error, r2_score, explained_variance_score, mean_absolute_error
from keras.models import Sequential
from keras.layers.core import Dense, Dropout, Activation
from keras.optimizers import Adam
from sklearn.model_selection import train_test_split
from keras.models import Sequential
from keras.layers.recurrent import LSTM
from keras.layers.core import Dense
from keras.utils import plot_model
#keras实现神经网络回归模型

# # 读取数据
# path = 'data.csv'
# train_df = pd.read_csv(path)
# # # 删掉不用字符串字段
# # dataset = train_df.drop('jh', axis=1)
# # df转array
# values = train_df.values
# # 原始数据标准化，为了加速收敛
# scaler = MinMaxScaler(feature_range=(0, 1))
# scaled = scaler.fit_transform(values)
# y = scaled[:, -1]
# X = scaled[:, 0:-1]
#
# # 划分训练集与测试集
# train_X, test_X, train_y, test_y = train_test_split(X, y, test_size=0.20, random_state=100)
# test_X1 = test_X
# # reshape为 3D [samples, timesteps, features]，将n_hours看成n个独立的时间序列而不是一个整体的
# train_X = train_X.reshape((train_X.shape[0], 1, train_X.shape[1]))
# test_X = test_X.reshape((test_X.shape[0], 1, test_X.shape[1]))
# print(train_X.shape, train_y.shape, test_X.shape, test_y.shape)


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
test_X1 = x_test
print('x_train_shape:{}'.format(X_train.shape))
print('x_test_shape:{}'.format(x_test.shape))
print('y_train_shape:{}'.format(Y_train.shape))
print('y_test_shape:{}'.format(y_test.shape))
print('\n')

# reshape为 3D [samples, timesteps, features]，将n_hours看成n个独立的时间序列而不是一个整体的
X_train = X_train.reshape((X_train.shape[0], 1, X_train.shape[1]))
x_test = x_test.reshape((x_test.shape[0], 1, x_test.shape[1]))
print(X_train.shape, Y_train.shape, x_test.shape, y_test.shape)

seed = 7
np.random.seed(seed)

from keras.layers import SimpleRNN
model = Sequential()
model.add(LSTM(units=100, input_shape=(X_train.shape[1], X_train.shape[2]), return_sequences=True))
model.add(Dropout(0.25))
model.add(LSTM(units=100))
model.add(Dropout(0.25))
model.add(Dense(1))
model.compile(loss='mean_squared_error', optimizer='adam')

plot_model(model, to_file='./model_LSTM.png', show_shapes=True)
history = model.fit(X_train, Y_train, epochs=500, batch_size=20, verbose=1)
predicts = model.predict(x_test)


# # loss曲线
# pyplot.plot(history.history['loss'], label='train')
# # pyplot.plot(history.history['val_loss'], label='test')
# pyplot.xlabel('epochs')    # x轴标题
# pyplot.ylabel('loss')  # y轴标题
# pyplot.legend()
# pyplot.show()

# # 预测y逆标准化
# yhat = model.predict(X_train)
# inv_yhat0 = np.concatenate((test_X1, yhat), axis=1)
# inv_yhat1 = scaler.inverse_transform(inv_yhat0)
# inv_yhat = inv_yhat1[:, -1]
#
# # 原始y逆标准化
# test_y = test_y.reshape((len(test_y), 1))
# inv_y0 = np.concatenate((test_X1, test_y), axis=1)
# inv_y1 = scaler.inver
#
# inv_y1 = scaler.inverse_transform(inv_y0)
# inv_y = inv_y1[:, -1]

# # 计算预测性能
# #print("Test score：", sqrt(r2_score(inv_y, inv_yhat)))
# print("Test MAE：", mean_absolute_error(inv_y, inv_yhat))
# print("Test RMSE：", sqrt(mean_squared_error(inv_y, inv_yhat)))
# print("score R2_score：", r2_score(inv_y, inv_yhat))
#
# #绘图表示预测值和实际值
# pyplot.plot(inv_y, label='true')
# pyplot.plot(inv_yhat, label='predicted')
# pyplot.legend()
# pyplot.show()

# ############################
# # 评估模型分数
# from sklearn import metrics
# from sklearn.model_selection import cross_val_score
#
# # 十折交叉验证，打印得分
# test_mse_scores = cross_val_score(model, x_test, y_test, cv=10, scoring='neg_mean_squared_error')
# test_mae_scores = cross_val_score(model, x_test, y_test, cv=10, scoring='neg_mean_absolute_error')
# test_r2_scores = cross_val_score(model, x_test, y_test, cv=10, scoring='r2')
# if test_r2_scores.mean() >= 0:
#     test_r_scores = np.sqrt(test_r2_scores.mean())
#     print('RNN_R: %.6f \n' % test_r_scores)
# print('RNN_MSE: %.6f, MAE: %.6f, R^2: %.6f \n' % (test_mse_scores.mean(), test_mae_scores.mean(), test_r2_scores.mean()))
# ############################

# 评估模型分数
import numpy as np
from sklearn import metrics
MSE = metrics.mean_squared_error(y_test, predicts)
print('MSE:', MSE)
RMSE = np.sqrt(metrics.mean_squared_error(y_test, predicts))
print('RMSE:', RMSE)
R2 = metrics.r2_score(y_test, predicts)
print("R^2:", R2)
# if R2 >= 0:
#     R = np.sqrt(metrics.r2_score(y_test, predicts))
#     print("R:", R)

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
plt.title("LSTM")

plt.scatter(y_test, predicts)
plt.xlabel("True Values")
plt.ylabel("Predictions")
plt.plot(y_test, y_test, color='r', linewidth=1.0, linestyle='-.', label='y = x')
plt.text(2, 23, 'Corr=' + str(round(Corr, 3)))
# plt.legend()
ax = plt.gca()
ax.set_aspect('equal')
plt.show()


