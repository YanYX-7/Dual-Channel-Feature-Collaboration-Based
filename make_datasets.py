# VMD + FFT 数据集的制作
import pandas as pd
import numpy as np
from vmdpy import VMD
import torch
from joblib import dump, load
import os
os.environ['KMP_DUPLICATE_LIB_OK']='True'

# -----VMD 参数--------
alpha = 2000
tau = 0
DC = 0
init = 1
tol = 1e-7
# -----参数--------


# VMD + FFT 数据集的制作
def make_feature_datasets(data, imfs_k):
    '''
        参数 data: 待分解数据
            imfs_k: VMD分解模态（IMF）个数

        返回 features:  提取的特征数据集
            y_label    : 数据集对应标签
    '''
    samples = data.shape[0]
    signl_len = data.shape[1]
    # 把数据转为numpy
    data = np.array(data)

    # 特征数量 分解数量  + 快速傅里叶变换结果
    features_num = imfs_k + 1

    # 构造三维矩阵
    features = np.zeros((samples, features_num, signl_len))

    # 对数据进行VMD分解 和 FFT 变换
    for i in range(samples):

        # Apply VMD
        u, u_hat, omega= VMD(data[i], alpha, tau, imfs_k, DC, init, tol)

        # 快速傅里叶变换
        fft_result1 = np.fft.fft(data[i])
        magnitude_spectrum = np.abs(fft_result1)
        magnitude_spectrum = magnitude_spectrum.reshape(1, -1)

        # 垂直叠加
        combined_matrix = np.vstack((u, magnitude_spectrum))
        features[i] = combined_matrix

    # 把numpy转为  tensor
    features = torch.tensor(features).float()
    return features



if __name__ == '__main__':

    # VMD分解: 使所有信号的分量特征保持同样的维度，K = 4
    # 加载数据
    train_xdata = load('./dataresult/trainX_1024_10c')
    val_xdata = load('./dataresult/valX_1024_10c')
    test_xdata = load('./dataresult/testX_1024_10c')
    train_ylabel = load('./dataresult/trainY_1024_10c')
    val_ylabel = load('./dataresult/valY_1024_10c')
    test_ylabel = load('./dataresult/testY_1024_10c')

    # VMD分解预处理  统一保存4个分量
    # 模态数量  分解模态（IMF）个数
    K = 4

    train_features = make_feature_datasets(train_xdata, K)
    val_features = make_feature_datasets(val_xdata, K)
    test_features = make_feature_datasets(test_xdata, K)

    # 保存数据
    dump(train_features, './dataresult/train_features_1024_10c')
    dump(val_features, './dataresult/val_features_1024_10c')
    dump(test_features, './dataresult/test_features_1024_10c')

    print('数据 形状：')
    print(train_features.shape, train_ylabel.shape)
    print(val_features.shape, val_ylabel.shape)
    print(test_features.shape, test_ylabel.shape)