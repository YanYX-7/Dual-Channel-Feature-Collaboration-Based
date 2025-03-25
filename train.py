import torch
from joblib import dump, load
import torch.nn as nn
import numpy as np
import time
import torch.utils.data as Data
import torch.nn.functional as F
import matplotlib.pyplot as plt
import matplotlib
matplotlib.rc("font", family='Microsoft YaHei')

import warnings
from matplotlib import MatplotlibDeprecationWarning
warnings.filterwarnings("ignore", category=MatplotlibDeprecationWarning)

# 参数与配置
torch.manual_seed(100)  # 设置随机种子，以使实验结果具有可重复性
device = torch.device("cuda" if torch.cuda.is_available() else "cpu") # 有GPU先用GPU训练


from models.model import CNNInformerMATTModel

# 加载数据集
def dataloader(batch_size, workers=2):
    # 训练集
    train_xdata = load('./dataresult/train_features_1024_10c')
    train_ylabel = load('./dataresult/trainY_1024_10c')
    # 验证集
    val_xdata = load('./dataresult/val_features_1024_10c')
    val_ylabel = load('./dataresult/valY_1024_10c')
    # 测试集
    test_xdata = load('./dataresult/test_features_1024_10c')
    test_ylabel = load('./dataresult/testY_1024_10c')

    # 加载数据
    train_loader = Data.DataLoader(dataset=Data.TensorDataset(train_xdata, train_ylabel),
                                   batch_size=batch_size, shuffle=True, num_workers=workers, drop_last=True)
    val_loader = Data.DataLoader(dataset=Data.TensorDataset(val_xdata, val_ylabel),
                                 batch_size=batch_size, shuffle=True, num_workers=workers, drop_last=True)
    test_loader = Data.DataLoader(dataset=Data.TensorDataset(test_xdata, test_ylabel),
                                  batch_size=batch_size, shuffle=True, num_workers=workers, drop_last=True)
    return train_loader, val_loader, test_loader


# 训练模型
def model_train(train_loader, val_loader, parameter):
    '''
          参数
          train_loader：训练集
          test_loader：测试集
          parameter： 参数
          返回
      '''
    device = parameter['device']
    model = parameter['model']
    model = model.to(device)
    # 参数
    epochs = parameter['epochs']
    learn_rate = parameter['learn_rate']

    # 定义损失函数和优化函数
    loss_function = nn.CrossEntropyLoss(reduction='sum')  # loss
    optimizer = torch.optim.Adam(model.parameters(), learn_rate)  # 优化器

    # 样本长度
    train_size = len(train_loader) * batch_size
    val_size = len(val_loader) * batch_size

    # 最高准确率  最佳模型
    best_accuracy = 0.0
    best_model = model

    train_loss = []  # 记录在训练集上每个epoch的loss的变化情况
    train_acc = []  # 记录在训练集上每个epoch的准确率的变化情况
    validate_acc = []
    validate_loss = []

    print('*' * 20, '开始训练', '*' * 20)
    # 计算模型运行时间
    start_time = time.time()
    for epoch in range(epochs):
        # 训练
        model.train()

        loss_epoch = 0.  # 保存当前epoch的loss和
        correct_epoch = 0  # 保存当前epoch的正确个数和
        for seq, labels in train_loader:
            seq, labels = seq.to(device), labels.to(device)
            # print(seq.size(), labels.size()) torch.Size([32, 7, 1024]) torch.Size([32])
            # 每次更新参数前都梯度归零和初始化
            optimizer.zero_grad()
            # 前向传播
            y_pred = model(seq)  # torch.Size([16, 10])
            # 对模型输出进行softmax操作，得到概率分布
            probabilities = F.softmax(y_pred, dim=1)
            # 得到预测的类别
            predicted_labels = torch.argmax(probabilities, dim=1)
            # 与真实标签进行比较，计算预测正确的样本数量  # 计算当前batch预测正确个数
            correct_epoch += (predicted_labels == labels).sum().item()
            # 损失计算
            loss = loss_function(y_pred, labels)
            loss_epoch += loss.item()
            # 反向传播和参数更新
            loss.backward()
            optimizer.step()
        #     break
        # break
        # 计算准确率
        train_Accuracy = correct_epoch / train_size
        train_loss.append(loss_epoch / train_size)
        train_acc.append(train_Accuracy)
        print(f'Epoch: {epoch + 1:2} train_Loss: {loss_epoch / train_size:10.8f} train_Accuracy:{train_Accuracy:4.4f}')
        # 每一个epoch结束后，在验证集上验证实验结果。
        with torch.no_grad():
            # 将模型设置为评估模式
            model.eval()

            loss_validate = 0.
            correct_validate = 0
            for data, label in val_loader:
                data, label = data.to(device), label.to(device)
                pre = model(data)
                # 对模型输出进行softmax操作，得到概率分布
                probabilities = F.softmax(pre, dim=1)
                # 得到预测的类别
                predicted_labels = torch.argmax(probabilities, dim=1)
                # 与真实标签进行比较，计算预测正确的样本数量  # 计算当前batch预测正确个数
                correct_validate += (predicted_labels == label).sum().item()
                loss = loss_function(pre, label)
                loss_validate += loss.item()
            # print(f'validate_sum:{loss_validate},  validate_Acc:{correct_validate}')
            val_accuracy = correct_validate / val_size
            print(f'Epoch: {epoch + 1:2} val_Loss:{loss_validate / val_size:10.8f},  validate_Acc:{val_accuracy:4.4f}')
            validate_loss.append(loss_validate / val_size)
            validate_acc.append(val_accuracy)
            # 如果当前模型的准确率优于之前的最佳准确率，则更新最佳模型
            # 保存当前最优模型参数
            if val_accuracy > best_accuracy:
                best_accuracy = val_accuracy
                best_model = model  # 更新最佳模型的参数

    print(f'\nDuration: {time.time() - start_time:.0f} seconds')

    # 最后的模型参数
    last_model = model
    print('*' * 20, '训练结束', '*' * 20)
    print(f'\nDuration: {time.time() - start_time:.0f} seconds')
    print("best_accuracy :", best_accuracy)

    # 可视化
    # 创建训练损失、准确率图
    plt.figure(figsize=(14, 7), dpi=100)  # dpi 越大  图片分辨率越高，写论文的话 一般建议300以上设置
    plt.plot(range(epochs), train_loss, color='blue', marker='o', label='Train-loss')
    plt.plot(range(epochs), train_acc, color='green', marker='*', label='Train-accuracy')
    plt.plot(range(epochs), validate_loss, color='red', marker='+', label='Validate_loss')
    plt.plot(range(epochs), validate_acc, color='orange', marker='x', label='Validate_accuracy')

    plt.xlabel('迭代次数', fontsize=12)
    plt.ylabel('训练损失值-准确率', fontsize=12)
    plt.xticks(fontsize=10)
    plt.yticks(fontsize=10)
    plt.legend(fontsize=12)
    plt.title('CNN-Informer-MATT  training visualization', fontsize=16)
    # plt.show()  # 显示 lable

    plt.savefig('train_result', dpi=100)
    # 保存结果 方便 后续画图处理（如果有需要的话）
    dump(train_loss, './dataresult/train_loss')
    dump(train_acc, './dataresult/train_acc')
    dump(validate_loss, './dataresult/validate_loss')
    dump(validate_acc, './dataresult/validate_acc')

    return last_model, best_model


if __name__ == '__main__':

    batch_size = 32
    # 加载数据
    train_loader, val_loader, test_loader = dataloader(batch_size)
    # 保存测试集数据， 后面进行测试
    dump(test_loader, './dataresult/test_loader')

    # 模型 参数设置
    # 1D-CNN 参数
    # VGG11，VGG13，VGG16，VGG19 可自行更换。
    # conv_arch = ((1, 64), (1, 128), (2, 256), (2, 512), (2, 512))  # vgg11
    # conv_arch = ((2, 64), (2, 128), (2 , 256), (2, 512), (2, 512))  # vgg13
    # conv_arch = ((2, 64), (2, 128), (3, 256), (3, 512), (3, 512))  # vgg16
    # conv_arch = ((2, 64), (2, 128), (4, 256), (4, 512), (4, 512))  # vgg19
    # 先用浅层试一试 1024      512      256       128        64     (32, 128, 64)
    conv_archs = ((1, 32), (1, 64), (1, 128), (1, 128))
    cnn_input_dim = 5  #  cnn输入维度数 为 5

    # 注意：这里是 把数据进行了堆叠 把一个5*1024 的矩阵 进行 划分堆叠成形状为 (5*8) * 128 就使输入序列的长度降下来了
    # 编码器输入大小（
    enc_in = 40  # 输入维度 把数据进行堆叠 5*1024 ->  40 * 128
    # 输出数据维度大小 10分类
    c_out = 10
    #  Probesparse attn因子（默认为5）
    factor = 5
    # 模型维度
    d_model = 128            # !!!
    # 多头注意力头数
    n_heads = 2   # 2 or 4
    # 编码器层数 默认 2 层
    e_layers = 2            # !!!
    # 模型中全连接网络（FCN）的维度
    d_ff = 200              # !!!
    # dropout概率
    dropout = 0.5

    # 一下参数默认即可
    # 编码器中使用的注意事项（默认为prob） 默认为"prob"论文的主要改进点，提出的注意力机制
    attn='prob'
    # 激活函数（默认为gelu）
    activation='gelu'
    output_attention = False
    distil = True

    # 定义 InformerSENetModel 模型
    model = CNNInformerMATTModel(conv_archs, cnn_input_dim, enc_in, c_out, factor, d_model, n_heads, e_layers, d_ff,
                 dropout, attn, activation, output_attention, distil)

    # 训练 参数设置
    learn_rate = 0.0003  # 学习率
    epochs = 50

    # 制作参数字典
    parameter = {
        'model': model,
        'epochs': epochs,
        'learn_rate': learn_rate,
        'device':device
    }

    # 训练模型
    last_model, best_model = model_train(train_loader, test_loader, parameter)
    # 保存最后的参数
    # torch.save(last_model, 'final_model_cnn_informer_matt.pt')
    # 保存最好的参数
    torch.save(best_model, 'best_model_cnn_informer_matt.pt')