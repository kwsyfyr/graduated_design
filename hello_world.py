##1-数据准备
#导入sklearn.datasets的load_boston数据集
from sklearn.datasets import load_boston
#导入的数据集是一种Bunch对象，它包括键keys和数值values,它有点类似字典，可用类似字段的方法查看信息
#获取字典的信息-获取字典的键dict.keys(),获取字典的值-dict.values(),获取字典的键值-dict.items(),获取特定键的值dict['键名']
from sklearn.svm import SVR

data = load_boston()
#获取字典的键dict.keys()
print(data.keys())    #该数据集跟之前的酒数据集一样，包含数据，目标分类、分类名，详细信息，数据的特征名，文件位置
print('\n')
#获取特定键的值dict['键名']
print('data的特征：',data['feature_names'])
print('\n')
print('data的分类：',data['target'])

##2-数据建模
# 2.1将数据拆分为训练集和测试集---要用train_test_split模块中的train_test_split()函数，随机将75%数据化道训练集，25%数据到测试集
# 导入数据集拆分工具
from sklearn.model_selection import train_test_split

# 将数据集的数值和分类目标赋值给x,y
x, y = data['data'], data['target']
# 拆分数据集---x,y都要拆分，rain_test_split(x,y,random_state=0),random_state=0使得每次生成的伪随机数不同
x_train, x_test, y_train, y_test = train_test_split(x, y, random_state=0)
# 查看拆分后的数据集大小情况
print('x_train_shape:{}'.format(x_train.shape))
print('x_test_shape:{}'.format(x_test.shape))
print('y_train_shape:{}'.format(y_train.shape))
print('y_test_shape:{}'.format(y_test.shape))
print('\n')

##2.2 模型训练/测试
# 用for函数一次性完成多个模型训练/测试--算法.fit(x_train,y_train)/算法.score(x_test,y_test)
for kernel in ['linear', 'rbf']:
    # 使用算法
    svr = SVR(kernel=kernel)
    # 算法.fit(x,y)对训练数据进行拟合
    svr.fit(x_train, y_train)
    # 打印拟合过程参数
    print(svr.fit(x_train, y_train))
    # 打印训练集得分
    print(kernel, '核函数的模型训练集得分：{:.3f}'.format(
        svr.score(x_train, y_train)))
    # 打印测试集得分
    print(kernel, '核函数的模型测试集得分：{:.3f}'.format(
        svr.score(x_test, y_test)))

    ##数据预处理
    # 查看一下各个特征的数量级情况（最大最小值）
    import numpy as np
    import matplotlib.pyplot as plt
    plt.plot(x.min(axis=0), 'v', label='min')
    plt.plot(x.max(axis=0), '^', label='max')
    plt.yscale('log')
    plt.legend(loc='best')
    plt.xlabel('features')
    plt.ylabel('feature magnitude')
    plt.show()

    ##数据预处理
    # 数据标准化
    from sklearn.preprocessing import StandardScaler

    scaler = StandardScaler()
    scaler.fit(x_train)
    X_train_scaled = scaler.transform(x_train)
    X_test_scaled = scaler.transform(x_test)
    plt.plot(X_train_scaled.min(axis=0), 'v', label='train set min')
    plt.plot(X_train_scaled.max(axis=0), '^', label='train set max')
    plt.plot(X_test_scaled.min(axis=0), 'v', label='test set min')
    plt.plot(X_test_scaled.max(axis=0), '^', label='test set max')
    plt.yscale('log')
    plt.legend(loc='best')
    plt.xlabel('scaled features')
    plt.ylabel('scaled feature magnitude')
    plt.show()

    # 使用预处理后的数据再来训练一次模型
    # 用for函数一次性完成多个模型训练/测试--算法.fit(x_train,y_train)/算法.score(x_test,y_test)
    for kernel in ['linear', 'rbf']:
        # 使用算法
        svr = SVR(kernel=kernel)
        # 算法.fit(x,y)对训练数据进行拟合
        svr.fit(X_train_scaled, y_train)
        # 打印拟合过程参数
        print(svr.fit(X_train_scaled, y_train))
        # 打印训练集得分
        print(kernel, '核函数的模型训练集得分：{:.3f}'.format(
            svr.score(X_train_scaled, y_train)))
        # 打印测试集得分
        print(kernel, '核函数的模型测试集得分：{:.3f}'.format(
            svr.score(X_test_scaled, y_test)))

        # 调参
        svr = SVR(C=100, gamma=0.1)
        svr.fit(X_train_scaled, y_train)
        print('调节参数后的模型在训练集得分：{:.3f}'.format(
            svr.score(X_train_scaled, y_train)))
        print('调节参数后的模型在测试集得分：{:.3f}'.format(
            svr.score(X_test_scaled, y_test)))