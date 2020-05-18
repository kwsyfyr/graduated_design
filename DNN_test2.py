import numpy
import pandas
from keras.models import Sequential
from keras.layers import Dense
from keras.utils import plot_model
from keras.wrappers.scikit_learn import KerasRegressor
from sklearn.model_selection import cross_val_score
from sklearn.model_selection import KFold
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import Pipeline

# # load dataset
# dataframe = pandas.read_csv("housing.csv", delim_whitespace=True, header=None)
# dataset = dataframe.values
# # split into input (X) and output (Y) variables
# X = dataset[:, 0:13]
# Y = dataset[:, 13]

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

# define base mode
def baseline_model():
    # create model
    model = Sequential()
    model.add(Dense(20, input_dim=300, kernel_initializer='normal', activation='relu'))
    model.add(Dense(1, kernel_initializer='normal'))
    # Compile model
    model.compile(loss='mean_squared_error', optimizer='adam')
    plot_model(model, to_file='./model_DNN2.png', show_shapes=True)
    return model

# fix random seed for reproducibility
seed = 7
numpy.random.seed(seed)
# evaluate model with standardized dataset
estimators = []
estimators.append(('standardize', StandardScaler()))
estimators.append(('mlp', KerasRegressor(build_fn=baseline_model, nb_epoch=100, batch_size=5, verbose=0)))
pipeline = Pipeline(estimators)
# use 10-fold cross validation to evaluate this baseline model
kfold = KFold(n_splits=10, random_state=seed)
results = cross_val_score(pipeline, X_train, Y_train, cv=kfold)
print("Standardized: %.2f (%.2f) MSE" % (results.mean(), results.std()))
###############3
pred = pipeline.predict(y_test)
# 画画预测图
import matplotlib.pyplot as plt

plt.title('DNN2')
plt.scatter(y_test, pred)
plt.xlabel("True Values")
plt.ylabel("Predictions")
plt.plot(y_test, y_test, color='r')
plt.show()