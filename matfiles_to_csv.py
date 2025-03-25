import numpy as np
import pandas as pd
from scipy.io import loadmat
import sklearn
import torch
from joblib import dump, load

# 时间步长 1024 和 重叠率 -0.5
# window = 1024  step = 512
# 切割划分方法: 参考论文 《时频图结合深度神经网络的轴承智能故障诊断研究》

def split_data_with_overlap(data, time_steps, lable, overlap_ratio=0.5):
    """
        data:要切分的时间序列数据,可以是一个一维数组或列表。
        time_steps:切分的时间步长,表示每个样本包含的连续时间步数。
        lable: 表示切分数据对应 类别标签
        overlap_ratio:前后帧切分时的重叠率,取值范围为 0 到 1,表示重叠的比例。
    """
    stride = int(time_steps * (1 - overlap_ratio))  # 计算步幅
    samples = (len(data) - time_steps) // stride + 1  # 计算样本数
    # 用于存储生成的数据
    Clasiffy_dataFrame = pd.DataFrame(columns=[x for x in range(time_steps + 1)])
    # 记录数据行数(量)
    index_count = 0
    data_list = []
    for i in range(samples):
        start_idx = i * stride
        end_idx = start_idx + time_steps
        temp_data = data[start_idx:end_idx].tolist()
        temp_data.append(lable)  # 对应哪一类
        data_list.append(temp_data)
    Clasiffy_dataFrame = pd.DataFrame(data_list, columns=Clasiffy_dataFrame.columns)
    return Clasiffy_dataFrame

# 归一化数据
def normalize(data):
    ''' (0,1)归一化
        参数:一维时间序列数据
    '''
    s= (data-min(data)) / (max(data)-min(data))
    return  s

# 数据集的制作
def make_datasets(dataframe, split_rate = [0.7,0.2,0.1]):
    '''
        参数:
        dataframe: 故障分类的数据集,csv格式,数据形状: 119808行  10列
        label_list: 故障分类标签
        split_rate: 训练集、验证集、测试集划分比例

        返回:
        train_set: 训练集数据
        val_set: 验证集数据
        test_set: 测试集数据
    '''
    # 1.读取数据
    origin_data = dataframe
    # 2.分割样本点
    time_steps = 1024  # 时间步长
    overlap_ratio = 0.5  # 重叠率
    # 用于存储生成的数据# 10个样本集合
    samples_data = pd.DataFrame(columns=[x for x in range(time_steps + 1)])
    # 记录类别标签
    label = 0
    # 使用iteritems()方法遍历每一列
    for column_name, column_data in origin_data.items():
        # 对数据集的每一维进行归一化
        # column_data = normalize(column_data)
        # 划分样本点  window = 512  overlap_ratio = 0.5  samples = 467 每个类有467个样本
        split_data = split_data_with_overlap(column_data, time_steps, label, overlap_ratio)
        label += 1 # 类别标签递增
        samples_data = pd.concat([samples_data, split_data])

    # 读取 csv 文件！
    # 1.读取数据
    # samples_data = pd.read_csv(data_file_csv)
    samples_data = samples_data.sample(frac=1).reset_index(drop=True)
    # 随机打乱样本点顺序
    # 打乱索引并重置索引
    samples_data = samples_data.sample(frac=1).reset_index(drop=True)
    # # 3.分割训练集-、验证集、测试集
    sample_len = len(samples_data) # 每一类样本数量
    train_len = int(sample_len*split_rate[0])  # 向下取整
    val_len = int(sample_len*split_rate[1])
    train_set = samples_data.iloc[0:train_len,:]
    val_set = samples_data.iloc[train_len:train_len+val_len,:]
    test_set = samples_data.iloc[train_len+val_len:sample_len,:]
    return  train_set, val_set, test_set



# 制作数据集和标签
# 这些转换是为了将数据和标签从Pandas数据结构转换为PyTorch可以处理的张量，
# 以便在神经网络中进行训练和预测。

def make_data_labels(dataframe):
    '''
        参数 dataframe: 数据框
        返回 x_data: 数据集     torch.tensor
            y_label: 对应标签值  torch.tensor
    '''
    # 信号值
    x_data = dataframe.iloc[:,0:-1]
    # 标签值
    y_label = dataframe.iloc[:,-1]
    x_data = torch.tensor(x_data.values).float()
    y_label = torch.tensor(y_label.values.astype('int64')) # 指定了这些张量的数据类型为64位整数，通常用于分类任务的类别标签
    return x_data, y_label



if __name__ == '__main__':

    # 文件名称
    file_names = ['0_0.mat', '7_1.mat', '7_2.mat', '7_3.mat', '14_1.mat', '14_2.mat', '14_3.mat', '21_1.mat',
                  '21_2.mat', '21_3.mat']

    # 采用驱动端数据
    data_columns = ['X097_DE_time', 'X105_DE_time', 'X118_DE_time', 'X130_DE_time', 'X169_DE_time',
                    'X185_DE_time', 'X197_DE_time', 'X209_DE_time', 'X222_DE_time', 'X234_DE_time']
    columns_name = ['de_normal', 'de_7_inner', 'de_7_ball', 'de_7_outer', 'de_14_inner', 'de_14_ball', 'de_14_outer',
                    'de_21_inner', 'de_21_ball', 'de_21_outer']

    data_12k_10c = pd.DataFrame()
    for index in range(10):
        # 读取MAT文件
        data = loadmat(f'matfiles\\{file_names[index]}')
        dataList = data[data_columns[index]].reshape(-1)
        data_12k_10c[columns_name[index]] = dataList[:119808]  # 121048  min: 121265
    print(data_12k_10c.shape)


    # 生成数据集
    train_set, val_set, test_set = make_datasets(data_12k_10c)
    # 制作标签
    train_xdata, train_ylabel = make_data_labels(train_set)
    val_xdata, val_ylabel = make_data_labels(val_set)
    test_xdata, test_ylabel = make_data_labels(test_set)

    # 保存数据
    dump(train_xdata, './dataresult/trainX_1024_10c')
    dump(val_xdata, './dataresult/valX_1024_10c')
    dump(test_xdata, './dataresult/testX_1024_10c')
    dump(train_ylabel, './dataresult/trainY_1024_10c')
    dump(val_ylabel, './dataresult/valY_1024_10c')
    dump(test_ylabel, './dataresult/testY_1024_10c')

    print('数据 形状：')
    print(train_xdata.size(), train_ylabel.shape)
    print(val_xdata.size(), val_ylabel.shape)
    print(test_xdata.size(), test_ylabel.shape)

