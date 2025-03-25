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


# 原始 测试集 进行 t-SNE 降维
def plot_orignal_TSNE(test_original_features, test_original_labels, num_classes, class_labels_name):
    features = np.array(test_original_features)
    labels = np.array(test_original_labels)

    # 初始化 t-SNE 模型
    tsne = TSNE(n_components=2, random_state=42)

    # 对原始数据进行 t-SNE 降维
    features_tsne = tsne.fit_transform(features)

    # 可视化结果
    plt.figure(figsize=(10, 8), dpi=100)
    # 绘制每个类别的散点图，并指定label
    for i in range(num_classes):
        plt.scatter(features_tsne[labels == i, 0], features_tsne[labels == i, 1], label=class_labels_name[i], alpha=0.8)
    plt.title('原始数据测试集 TSNE 可视化')
    plt.xlabel('tSNE Dimension 1')
    plt.ylabel('tSNE Dimension 2')
    # 显示legend
    plt.legend()
    # plt.show()
    plt.savefig('原始测试集TSNE可视化', dpi=100)

# 训练好的模型 测试集 进行 t-SNE 降维
def plot_model_TSNE(test_model_features, test_original_labels, num_classes, class_labels_name):
    features = np.array(test_model_features)
    labels = np.array(test_original_labels)

    # 初始化 t-SNE 模型
    tsne = TSNE(n_components=2, random_state=42)

    # 对原始数据进行 t-SNE 降维
    features_tsne = tsne.fit_transform(features)

    # 可视化结果
    plt.figure(figsize=(10, 8), dpi=100)
    # 绘制每个类别的散点图，并指定label
    for i in range(num_classes):
        plt.scatter(features_tsne[labels == i, 0], features_tsne[labels == i, 1], label=class_labels_name[i], alpha=0.8)
    plt.title('模型测试集 TSNE 可视化')
    plt.xlabel('tSNE Dimension 1')
    plt.ylabel('tSNE Dimension 2')
    # 显示legend
    plt.legend()
    # plt.show()
    plt.savefig('模型测试集TSNE可视化', dpi=100)

# 标签真实值和预测值对比
def plot_true_pre_compare(true_labels, predicted_labels, class_labels):
    # 将真实标签和预测标签转换为标签名称
    y_real_labels = [class_labels[label] for label in true_labels]
    y_pre_labels = [class_labels[label] for label in predicted_labels]

    # 对真实标签和预测标签进行排序
    sorted_indices = np.argsort(true_labels)
    y_real_sorted = [y_real_labels[i] for i in sorted_indices]
    y_pre_sorted = [y_pre_labels[i] for i in sorted_indices]

    # 可视化结果
    plt.figure(figsize=(10, 6), dpi=100)
    # 计算样本索引
    index = np.arange(len(y_real_sorted))

    # 绘制直线图
    plt.plot(index, y_real_sorted, color='green', marker='o', linestyle='None', label='真实标签')
    plt.plot(index, y_pre_sorted, color='orange', marker='x', linestyle='None', label='预测标签')

    plt.title('测试集真实标签与预测标签对比图')
    plt.xlabel('样本个数')
    plt.ylabel('标签')

    # 每隔一定数量的样本显示一个标签
    step = max(1, len(y_real_sorted) // 10)  # 每10个样本显示一个标签
    plt.xticks(ticks=index[::step], labels=index[::step], rotation=90)  # 这里改为使用index作为显示标签
    plt.legend()
    plt.grid(True)
    # plt.show()
    plt.savefig('标签真实值和预测值对比', dpi=100)


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

    # 混淆矩阵
    # 原始标签和自定义标签的映射
    label_mapping = {
        0: "C1", 1: "C2", 2: "C3", 3: "C4", 4: "C5",
        5: "C6", 6: "C7", 7: "C8", 8: "C9", 9: "C10",
    }
    confusion_mat = confusion_matrix(true_labels, predicted_labels)

    # 绘制混淆矩阵
    plt.figure(figsize=(10, 8), dpi=300)
    sns.heatmap(confusion_mat, xticklabels=label_mapping.values(), yticklabels=label_mapping.values(), annot=True,
                fmt='d', cmap='viridis')
    # plt.xlabel('Predicted Labels')
    # plt.ylabel('True Labels')
    # plt.title('Confusion Matrix')
    # plt.show()
    plt.savefig('Confusion_Matrix', dpi=100)

    num_classes = 10  # 10分类
    # 类别标签
    class_labels = ['de_normal', 'de_7_inner', 'de_7_ball', 'de_7_outer', 'de_14_inner',
                    'de_14_ball', 'de_14_outer', 'de_21_inner', 'de_21_ball', 'de_21_outer']

    # 训练好的模型 测试集 进行 t-SNE 降维
    plot_model_TSNE(test_model_features, true_labels, num_classes, class_labels)

    # 绘制 标签真实值和预测值对比
    plot_true_pre_compare(true_labels, predicted_labels, class_labels)


    # 原始 测试集 进行 t-SNE 降维
    # 测试集   注意 这里加载 最原始 的数据集， 不加载预处理后的数据
    test_x = load('./dataresult/testX_1024_10c')
    test_y = load('./dataresult/testY_1024_10c')

    batch_size = 32

    # 加载数据
    testloader = Data.DataLoader(dataset=Data.TensorDataset(test_x, test_y),
                                 batch_size=batch_size, shuffle=True, num_workers=2, drop_last=True)
    test_original_features = []  # 原始测试集 特征

    for test_data, test_label in testloader:
        test_original_features += test_data.tolist()  # 提取特征

    plot_orignal_TSNE(test_original_features, true_labels, num_classes, class_labels)




