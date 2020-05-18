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
fregre_sele = SelectKBest(f_regression, k = 800)
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
# # 导入弹性网
# from sklearn.linear_model import ElasticNet
# # 实例化弹性网类，设定随机种子，保证每次计算结果都相同
# e_net = ElasticNet(l1_ratio=0.7)
# # 训练模型
# alphas = np.arange(0.0000001, 0.0000002, .00000001)
# train_errors = list()
# test_errors = list()
# for alpha in alphas:
#     e_net.set_params(alpha = alpha)
#     e_net.fit(x_train, y_train)
#     train_errors.append(e_net.score(x_train, y_train))
#     test_errors.append(e_net.score(x_test,y_test))
# i_alpha_optim = np.argmax(train_errors)
# alpha_optim = alphas[i_alpha_optim]
# print(("Optimal regularization parameter:%s" % alpha_optim))
# e_net.set_params(alpha= alpha_optim)

# 导入弹性网
from sklearn.linear_model import ElasticNet
# 实例化弹性网类，设定随机种子，保证每次计算结果都相同
# e_net = ElasticNet(alpha=0.9, l1_ratio=0.01)
e_net = ElasticNet(alpha=0.8, l1_ratio=0.01)
e_net.fit(x_train, y_train)
# print(e_net.alpha_)
# print(e_net.l1_ratio_)

# ############################################
# from sklearn.model_selection import GridSearchCV
# # 设置调优超参数范围
# tuned_parameters = [{'alpha':(0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9, 1), 'l1_ratio':(0.01, 0.011, 0.012, 0.013, 0.014, 0.015, 0.016, 0.017, 0.018, 0.019, 0.020)}]
# # 通过 GridSearchCV 搜索最佳超参数
# clf = GridSearchCV(e_net, tuned_parameters, cv=10, scoring='neg_mean_squared_error')
# clf.fit(x_train, y_train)
# cv_result = pd.DataFrame.from_dict(clf.cv_results_)
# with open('e_net_cv_result.csv','w') as f:
#     cv_result.to_csv(f)
# print('Results:')
# print(clf.cv_results_)
# print("最好参数为")
# print(clf.best_params_)
# print("Best svr:")
# print(clf.best_estimator_)
# print('Best score is:')
# print(clf.best_score_)
# pred = clf.predict(x_test)
# ###########################################

# print(e_net.score(x_train, y_train))
# # 十折交叉验证，打印得分
# from sklearn.model_selection import cross_val_score
# scores = cross_val_score(e_net, x_train, y_train, cv=10)
# print('Accuracy: %.3f +/- %.3f' % (scores.mean(), scores.std()))

# # 导入弹性网
# from sklearn.linear_model import ElasticNetCV
# e_net = ElasticNetCV(alphas=np.arange(0.5, 1, 0.1), l1_ratio=np.arange(0.010, 0.020, .001), max_iter=5000, cv=10)
# e_net.fit(x_train, y_train)
# print(e_net.alpha_)
# print(e_net.l1_ratio_)
# print(e_net.score(x_train, y_train))
# # 十折交叉验证，打印得分
# from sklearn.model_selection import cross_val_score
# scores = cross_val_score(e_net, x_train, y_train, cv=10)
# print('Accuracy: %.3f +/- %.3f' % (scores.mean(), scores.std()))

# pred 用于保存预测结果
pred = e_net.predict(x_test)
# print(e_net.score(x_test, y_test))

# 评估模型分数
from sklearn import metrics
from sklearn.model_selection import cross_val_score

# 十折交叉验证，打印得分
test_mse_scores = cross_val_score(e_net, x_test, y_test, cv=10, scoring='neg_mean_squared_error')
test_mae_scores = cross_val_score(e_net, x_test, y_test, cv=10, scoring='neg_mean_absolute_error')
test_r2_scores = cross_val_score(e_net, x_test, y_test, cv=10, scoring='r2')
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
#
# MSE = metrics.mean_squared_error(y_test, pred)
# RMSE = np.sqrt(metrics.mean_squared_error(y_test, pred))
# R2 = metrics.r2_score(y_test, pred)
# R = np.sqrt(metrics.r2_score(y_test, pred))
# print('MSE:', MSE)
# print('RMSE:', RMSE)
# print("R^2:", R2)
# print("R:", R)

# 画画图
import matplotlib.pyplot as plt
from scipy import optimize


def f_1(x, A, B):
    return A * x + B


A1, B1 = optimize.curve_fit(f_1, y_test, pred)[0]
nihe_x = y_test
nihe_y = A1 * nihe_x + B1
plt.plot(nihe_x, nihe_y, color='b', linewidth=2.0, label='Predicted Label')

plt.title('E_net')
plt.scatter(y_test, pred)
plt.xlabel("True Values")
plt.ylabel("Predictions")
plt.plot(y_train, y_train, color='r', linewidth=1.0, linestyle='-.', label='y = x')
plt.text(2, 23, 'Corr=' + str(round(Corr, 3)))
# plt.legend()
ax = plt.gca()
ax.set_aspect('equal')
plt.show()