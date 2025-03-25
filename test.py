from joblib import dump, load
import torch
import torch.nn as nn
import numpy as np
import matplotlib.pyplot as plt
import matplotlib
matplotlib.rc("font", family='Microsoft YaHei')

from sklearn.metrics import confusion_matrix
from sklearn.metrics import classification_report
import seaborn as sns
from sklearn.manifold import TSNE
import torch.utils.data as Data
import warnings
from matplotlib import MatplotlibDeprecationWarning
warnings.filterwarnings("ignore", category=MatplotlibDeprecationWarning)

torch.manual_seed(100)  # 设置随机种子，以使实验结果具有可重复性
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")


# 模型 测试集 测试
def model_test(model, test_loader):
    model = model.to(device)


    true_labels = []  # 存储类别标签
    predicted_labels = []  # 存储预测的标签

    # 提取特征
    test_model_features = []  # 测试集经过训练后的模型得到的 特征

    with torch.no_grad():
        for test_data, test_label in test_loader:
            # 将模型设置为评估模式
            model.eval()
            test_data, test_label = test_data.to(device), test_label.to(device)
            test_output = model(test_data)
            test_model_features += test_output.tolist()  # 提取特征

            # 计算个数
            predictedlabel = torch.argmax(test_output, dim=1)
            # 标签累计
            true_labels.extend(test_label.tolist())
            predicted_labels.extend(predictedlabel.tolist())

    return true_labels, predicted_labels, test_model_features


if __name__ == '__main__':

    # 加载模型
    model = torch.load('best_model_cnn_informer_matt.pt')
    # 加载测试集
    test_loader = load('./dataresult/test_loader')

    # 模型分类预测
    true_labels, predicted_labels, test_model_features = model_test(model, test_loader)

    # 计算每一类的分类准确率
    report = classification_report(true_labels, predicted_labels, digits=4)
    print(report)

   




