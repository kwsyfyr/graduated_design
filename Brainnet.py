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
target_csv = pd.read_csv(r"data\unrestricted_imei1_5_6_2019_20_9_17.csv", usecols=(0, 144))
value = target_csv.values

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
feature_X, targer_Y, feature_2dX = feature_mat['CorrVec'], feature_mat['target'], feature_mat['AA']
# 删除找到的空值
feature_X = np.delete(feature_X, [45, 523, 768], axis=0)
targer_Y = np.delete(targer_Y, [45, 523, 768], axis=0)
feature_2dX = feature_2dX.reshape(-1, 100, 100, 1)
feature_2dX = np.delete(feature_2dX, [45, 523, 768], axis=0)

##数据预处理
# 数据标准化，使用preprocessing库的StandardScaler类对数据进行标准化
from sklearn.preprocessing import StandardScaler

# 标准化，返回值为标准化后的数据
# feature_2dX = StandardScaler().fit_transform(feature_2dX)
# targer_Y = StandardScaler().fit_transform(targer_Y)
# 特征选择
from sklearn.feature_selection import SelectKBest
from scipy.stats import pearsonr
from sklearn.feature_selection import f_regression
# feature_X = np.squeeze(feature_X)
targer_Y = np.squeeze(targer_Y)
## use Pearson
# sele_X = SelectKBest(lambda X,Y: np.array(list(map(lambda x: pearsonr(x, Y)[0], X.T))).T, k = 300).fit_transform(feature_X, targer_Y)
# 使用 f_regression
# fregre_sele = SelectKBest(f_regression, k = 300)
# sele_2dX = fregre_sele.fit_transform(feature_2dX, targer_Y)
# 使用 mutal_info_regression 互信息
# from minepy import MINE
# def mic(x, y):
#     m = MINE()
#     m.compute_score(x, y)
#     return (m.mic(), 0.5)
# sele_X = SelectKBest(lambda X, Y: np.array(list(map(lambda x: mic(x, Y)[0], X.T))).T, k = 300).fit_transform(feature_X, targer_Y)

# 拆分数据集---x,y都要拆分，rain_test_split(x,y,random_state=0),random_state=0使得每次生成的伪随机数不同
X_train, x_test, Y_train, y_test = train_test_split(feature_2dX, targer_Y, random_state=0)
test_X1 = x_test
print('x_train_shape:{}'.format(X_train.shape))
print('x_test_shape:{}'.format(x_test.shape))
print('y_train_shape:{}'.format(Y_train.shape))
print('y_test_shape:{}'.format(y_test.shape))
print('\n')
# print(X_train.type)
# X_train = np.expand_dims(X_train, axis=-1)
# X_train = np.expand_dims(X_train, axis=-1)
# # x_test = np.expand_dims(x_test, axis=-1)
# x_test = np.expand_dims(x_test, axis=-1)
# X_train = X_train.reshape((X_train.shape[0], X_train.shape[1], 1))
# x_test = X_train.reshape((x_test.shape[0], x_test.shape[1], 1))
# print(X_train.shape)


# 使用keras构建模型
from keras.layers import Input, Dropout, Dense, Activation, LeakyReLU, Conv2D, Conv1D
from keras.layers import Lambda, Flatten
from keras.models import Model
from keras.layers.merge import add
from keras.layers.core import Permute
from keras import backend as K
from keras.regularizers import l2
from keras.layers.normalization import BatchNormalization

K.set_image_data_format('channels_last')
seed = 7
np.random.seed(seed)

def hcp_Edge2Edge(inputs, dim, filters, activation):
    row = Conv2D(filters, (1, dim), padding='valid', activation=None)(inputs)
    col = Conv2D(filters, (dim, 1), padding='valid', activation=None)(inputs)
    tiled_row = Lambda(lambda x: K.repeat_elements(x, rep=dim, axis=2),
                       (dim, dim, filters))(row)
    tiled_col = Lambda(lambda x: K.repeat_elements(x, rep=dim, axis=1),
                       (dim, dim, filters))(col)

    Sum = add([tiled_row, tiled_col])
    return activation(Sum)

def hcp_Edge2Node(inputs, dim, filters, activation):
    row = Conv2D(filters, (1, dim), padding='valid', activation=None)(inputs)
    col = Conv2D(filters, (dim, 1), padding='valid', activation=None)(inputs)

    Sum = add([row, Permute(dims=(2, 1, 3))(col)])
    return activation(Sum)

def hcp_Node2Graph(inputs, dim, filters, activation):
    nodes = Conv2D(filters, (dim, 1), padding='valid', activation=None)(inputs)

    return activation(nodes)

def hcp_brainnetcnn(dim, n_measure, e2e, e2n, n2g, dropout, leaky_alpha, nb_features=1):
    activation = LeakyReLU(alpha=leaky_alpha)
    In = Input(shape=(dim, dim, nb_features))
    layer_beta = hcp_Edge2Edge(In, dim, e2e, activation=activation)
    layer0 = hcp_Edge2Node(layer_beta, dim, e2n, activation=activation)
    layer1 = Dropout(dropout)(layer0)
    layer2 = hcp_Node2Graph(layer1, dim, n2g, activation=activation)
    layer3 = Flatten()(layer2)
    layer4 = Dense(n_measure, activation='linear')(layer3)
    return Model(inputs=In, outputs=layer4)

# print(X_train.shape[1])
model = hcp_brainnetcnn(X_train.shape[1], 1, 16, 128, 26, 0.5, 0.1)

# from keras.optimizers import SGD
# optimizer = SGD(lr=0.001, momentum=0.9, decay=0.0001, nesterov=False)
model.compile(loss='mean_squared_error', optimizer='adam')
plot_model(model, to_file='./model_Brainnet.png', show_shapes=True)

history = model.fit(X_train, Y_train, epochs=100, batch_size=64, verbose=1)
predicts = model.predict(x_test)
print(predicts)

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
plt.title("Brainnetcnn")

plt.scatter(y_test, predicts)
plt.xlabel("True Values")
plt.ylabel("Predictions")
plt.plot(y_test, y_test, color='r', linewidth=1.0, linestyle='-.', label='y = x')
# plt.legend()
plt.text(2, 23, 'Corr=' + str(round(Corr, 3)))
ax = plt.gca()
ax.set_aspect('equal')
plt.show()
