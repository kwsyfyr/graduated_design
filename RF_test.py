#author Feng
#utf-8

#导入标签数据集

import numpy as np
import pandas as pd
import os



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
fregre_sele = SelectKBest(f_regression, k = 100)
sele_X = fregre_sele.fit_transform(feature_X, targer_Y)
# 使用 mutal_info_regression 互信息
# from minepy import MINE
# def mic(x, y):
#     m = MINE()
#     m.compute_score(x, y)
#     return (m.mic(), 0.5)
# sele_X = SelectKBest(lambda X, Y: np.array(list(map(lambda x: mic(x, Y)[0], X.T))).T, k = 300).fit_transform(feature_X, targer_Y)

# 拆分数据集---x,y都要拆分，rain_test_split(x,y,random_state=0),random_state=0使得每次生成的伪随机数不同
x_train, x_test, y_train, y_test = train_test_split(sele_X, targer_Y, random_state=0)
print('x_train_shape:{}'.format(x_train.shape))
print('x_test_shape:{}'.format(x_test.shape))
print('y_train_shape:{}'.format(y_train.shape))
print('y_test_shape:{}'.format(y_test.shape))
print('\n')
# print(y_test)

##2.2 模型训练/测试
# 导入随机森林
from sklearn.ensemble import RandomForestRegressor
rf = RandomForestRegressor(n_estimators=10)
rf.fit(x_train, y_train)
# print(rf.score(x_train, y_train))
# # 十折交叉验证，打印得分
from sklearn.model_selection import cross_val_score
# scores = cross_val_score(rf, x_train, y_train, cv=10)
# print('Accuracy: %.3f +/- %.3f' % (scores.mean(), scores.std()))
# pred 用于保存预测结果
pred = rf.predict(x_test)
# print(rf.score(x_test, y_test))

# # 评估模型分数
from sklearn import  metrics
# MSE = metrics.mean_squared_error(y_test, pred)
# RMSE = np.sqrt(metrics.mean_squared_error(y_test, pred))
# R2 = metrics.r2_score(y_test, pred)
# R = np.sqrt(metrics.r2_score(y_test, pred))
# print('MSE:', MSE)
# print('RMSE:', RMSE)
# print("R^2:", R2)
# print("R:", R)

# 十折交叉验证，打印得分
test_mse_scores = cross_val_score(rf, x_test, y_test, cv=10, scoring='neg_mean_squared_error')
test_mae_scores = cross_val_score(rf, x_test, y_test, cv=10, scoring='neg_mean_absolute_error')
test_r2_scores = cross_val_score(rf, x_test, y_test, cv=10, scoring='r2')
if test_r2_scores.mean() >= 0:
    test_r_scores = np.sqrt(test_r2_scores.mean())
    print('R: %.6f \n' % test_r_scores)
print('MSE: %.6f, MAE: %.6f, R^2: %.6f \n' % (test_mse_scores.mean(), test_mae_scores.mean(), test_r2_scores.mean()))

data = pd.DataFrame(y_test)
data['2'] = pred
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


A1, B1 = optimize.curve_fit(f_1, y_test, pred)[0]
nihe_x = y_test
nihe_y = A1 * nihe_x + B1
plt.plot(nihe_x, nihe_y, color='b', linewidth=2.0, label='Predicted Label')
plt.title('RandomForest')
plt.scatter(y_test, pred)
plt.xlabel("True Values")
plt.ylabel("Predictions")
plt.plot(y_train, y_train, color='r', linewidth=1.0, linestyle='-.', label='y = x')
plt.text(2, 23, 'Corr=' + str(round(Corr, 3)))
# plt.legend()
ax = plt.gca()
ax.set_aspect('equal')
plt.show()