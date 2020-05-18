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
transfer = StandardScaler()
feature_X = transfer.fit_transform(feature_X)
# targer_Y = transfer.fit_transform(targer_Y)
# 特征选择
from sklearn.feature_selection import SelectKBest
from scipy.stats import pearsonr
from sklearn.feature_selection import f_regression
feature_X = np.squeeze(feature_X)
targer_Y = np.squeeze(targer_Y)
# # use Pearson
# sele_X = SelectKBest(lambda X,Y: np.array(list(map(lambda x: pearsonr(x, Y)[0], X.T))).T, k = 300).fit_transform(feature_X, targer_Y)
#
# 使用 f_regression
fregre_sele = SelectKBest(f_regression, k = 800)
sele_X = fregre_sele.fit_transform(feature_X, targer_Y)

# # 使用 mutal_info_regression 互信息
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
# 用for函数一次性完成多个模型训练/测试--算法.fit(x_train,y_train)/算法.score(x_test,y_test)
from sklearn.svm import SVR
from sklearn.model_selection import cross_val_score
from sklearn.model_selection import GridSearchCV

# rbf, sigmoid
for kernel in ['rbf']:
    # 使用算法
    svr = SVR(kernel=kernel, gamma='scale', C=20)
    # 算法.fit(x,y)对训练数据进行拟合
    model = svr.fit(x_train, y_train)
    # 打印拟合过程参数
    print(model.fit(x_train, y_train))
    pred = model.predict(x_test)
    # ############################################
    # # 设置调优超参数范围
    # tuned_parameters = [{'gamma':('scale', 1e-4, 1e-3, 1e-2, 1e-1, 1, 10), 'C':(1, 10, 20, 30, 40, 50, 60, 70, 80, 90, 100)}]
    # # 通过 GridSearchCV 搜索最佳超参数
    # clf = GridSearchCV(model, tuned_parameters, cv=10, scoring='neg_mean_squared_error')
    # clf.fit(x_train, y_train)
    # cv_result = pd.DataFrame.from_dict(clf.cv_results_)
    # with open(kernel + '_cv_result.csv','w') as f:
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

    # # 评估模型分数
    # from sklearn import metrics
    # MSE = metrics.mean_squared_error(y_test, pred)
    # RMSE = np.sqrt(metrics.mean_squared_error(y_test, pred))
    # R2 = metrics.r2_score(y_test, pred)
    # R = -1
    # if R2 >= 0:
    #     R = np.sqrt(metrics.r2_score(y_test, pred))
    # print('MSE:', MSE)
    # print('RMSE:', RMSE)
    # print("R^2:", R2)
    # print("R:", R)

    ############################
    # 评估模型分数
    from sklearn import metrics
    # 十折交叉验证，打印得分
    test_mse_scores = cross_val_score(model, x_test, y_test, cv=10, scoring='neg_mean_squared_error')
    test_mae_scores = cross_val_score(model, x_test, y_test, cv=10, scoring='neg_mean_absolute_error')
    test_r2_scores = cross_val_score(model, x_test, y_test, cv=10, scoring='r2')
    if test_r2_scores.mean() >= 0:
        test_r_scores = np.sqrt(test_r2_scores.mean())
        print(kernel, 'R: %.6f \n' % test_r_scores)
    print(kernel, 'MSE: %.6f, MAE: %.6f, R^2: %.6f \n' % (test_mse_scores.mean(), test_mae_scores.mean(), test_r2_scores.mean()))
    ############################
    # print('r2 is : %.3f' % (scores.r2()))
    # 打印训练集得分
    # print(kernel, '核函数的模型训练集得分：{:.3f}'.format(
    #     svr.score(x_train, y_train)))
    # # 打印测试集得分
    # print(kernel, '核函数的模型测试集得分：{:.3f}'.format(
    #     svr.score(x_test, y_test)))
    data = pd.DataFrame(y_test)
    data['2'] = pred
    # data.rename(columns={'0':'y_test', '2':'predicts'})
    data.columns = ['y_test', 'predicts']
    # print(data)
    Corr = data['y_test'].corr(data['predicts'])
    print("Corr:", Corr)

    ############################
    # 画画预测图
    import matplotlib.pyplot as plt
    from sklearn import linear_model
    from scipy import optimize
    def f_1(x, A, B):
        return A*x + B
    A1, B1 = optimize.curve_fit(f_1, y_test, pred)[0]
    nihe_x = y_test
    nihe_y = A1 * nihe_x + B1
    #plt.plot(nihe_x, nihe_y, 'blue')

    # LR = linear_model.LinearRegression()
    # LR.fit(y_test, pred)
    # print(LR.coef_, LR.intercept_)
    # nihe_x = y_test
    # nihe_pred_y = LR.coef_ * nihe_x + LR.intercept_

    plt.title(kernel)
    plt.scatter(y_test, pred)
    plt.xlabel("True Values")
    plt.ylabel("Predictions")
    plt.plot(nihe_x, nihe_y, color = 'b', linewidth=2.0, label='Predicted Label')
    plt.plot(y_test, y_test, color = 'r', label='y = x', linewidth=1.0, linestyle='-.')
    # plt.legend()
    plt.text(2, 23, 'Corr=' + str(round(Corr, 3)))
    ax = plt.gca()
    ax.set_aspect('equal')
    plt.show()

