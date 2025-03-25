import torch
import torch.nn as nn
import torch.nn.functional as F

from models.encoder import Encoder, EncoderLayer, ConvLayer
from models.attn import FullAttention, ProbAttention, AttentionLayer
from models.embed import DataEmbedding

# 定义多头注意力机制
import torch
import torch.nn as nn
import torch.nn.functional as F
import math
device = torch.device("cuda" if torch.cuda.is_available() else "cpu") # 有GPU先用GPU训练

def attention(query, key, value):
    """
    计算Attention的结果。
    这里其实传入的是Q,K,V，而Q,K,V的计算是放在模型中的，请参考后续的MultiHeadedAttention类。

    这里的Q,K,V有两种Shape，如果是Self-Attention，Shape为(batch, 词数, d_model)，
                           例如(1, 7, 128)，即batch_size为1，一句7个单词，每个单词128维

                           但如果是Multi-Head Attention，则Shape为(batch, head数, 词数，d_model/head数)，
                           例如(1, 8, 7, 16)，即Batch_size为1，8个head，一句7个单词，128/8=16。
                           这样其实也能看出来，所谓的MultiHead其实就是将128拆开了。

                           在Transformer中，由于使用的是MultiHead Attention，所以Q,K,V的Shape只会是第二种。

    """

    # 获取d_model的值。之所以这样可以获取，是因为query和输入的shape相同，
    # 若为Self-Attention，则最后一维都是词向量的维度，也就是d_model的值。
    # 若为MultiHead Attention，则最后一维是 d_model / h，h为head数
    d_k = query.size(-1)
    # 执行QK^T / √d_k
    scores = torch.matmul(query, key.transpose(-2, -1)) / math.sqrt(d_k)

    # 执行公式中的Softmax
    # 这里的p_attn是一个方阵
    # 若是Self Attention，则shape为(batch, 词数, 次数)，例如(1, 7, 7)
    # 若是MultiHead Attention，则shape为(batch, head数, 词数，词数)
    # 对scores的最后一维进行softmax操作，使用F.softmax方法，这样获得最终的注意力张量
    p_attn = F.softmax(scores, dim=-1)

    # 最后再乘以 V。
    # 对于Self Attention来说，结果Shape为(batch, 词数, d_model)，这也就是最终的结果了。
    # 但对于MultiHead Attention来说，结果Shape为(batch, head数, 词数，d_model/head数)
    # 而这不是最终结果，后续还要将head合并，变为(batch, 词数, d_model)。不过这是MultiHeadAttention
    # 该做的事情。
    return torch.matmul(p_attn, value)


class MultiHeadedAttention(nn.Module):
    def __init__(self, h, d_model):
        """
        h: head的数量
        """
        super(MultiHeadedAttention, self).__init__()
        assert d_model % h == 0
        # We assume d_v always equals d_k
        self.d_k = d_model // h
        self.h = h
        # 定义W^q, W^k, W^v和W^o矩阵。
        self.linears = [
            nn.Linear(d_model, d_model),
            nn.Linear(d_model, d_model),
            nn.Linear(d_model, d_model),
            nn.Linear(d_model, d_model),
        ]

    def forward(self, query, key, value):
        # 获取Batch Size
        nbatches = query.size(0)

        query = query.to(device)
        key = key.to(device)
        value = value.to(device)
        """
        1. 求出Q, K, V，这里是求MultiHead的Q,K,V，所以Shape为(batch, head数, 词数，d_model/head数)
            1.1 首先，通过定义的W^q,W^k,W^v求出SelfAttention的Q,K,V，此时Q,K,V的Shape为(batch, 词数, d_model)
                对应代码为 `linear(x)`
            1.2 分成多头，即将Shape由(batch, 词数, d_model)变为(batch, 词数, head数，d_model/head数)。
                对应代码为 `view(nbatches, -1, self.h, self.d_k)`
            1.3 最终交换“词数”和“head数”这两个维度，将head数放在前面，最终shape变为(batch, head数, 词数，d_model/head数)。
                对应代码为 `transpose(1, 2)`
        """

        # Ensure that the linear layers are also on the correct device
        self.linears = [linear.to(device) for linear in self.linears]

        query, key, value = [
            linear(x).view(nbatches, -1, self.h, self.d_k).transpose(1, 2).to(device)
            for linear, x in zip(self.linears, (query, key, value))
        ]

        """
        2. 求出Q,K,V后，通过attention函数计算出Attention结果，
           这里x的shape为(batch, head数, 词数，d_model/head数)
           self.attn的shape为(batch, head数, 词数，词数)
        """
        x = attention(
            query, key, value
        )

        """
        3. 将多个head再合并起来，即将x的shape由(batch, head数, 词数，d_model/head数)
           再变为 (batch, 词数，d_model)
           3.1 首先，交换“head数”和“词数”，这两个维度，结果为(batch, 词数, head数, d_model/head数)
               对应代码为：`x.transpose(1, 2).contiguous()`
           3.2 然后将“head数”和“d_model/head数”这两个维度合并，结果为(batch, 词数，d_model)
        """
        x = (
            x.transpose(1, 2)
                .contiguous()
                .view(nbatches, -1, self.h * self.d_k)
        )

        # 最终通过W^o矩阵再执行一次线性变换，得到最终结果。
        return self.linears[-1](x)

# CNN-Informer-MATT 分类模型
class CNNInformerMATTModel(nn.Module):
    def __init__(self, conv_archs, cnn_input_dim, enc_in, c_out, factor=5, d_model=512, n_heads=8, e_layers=3, d_ff=512,
                 dropout=0.0, attn='prob', activation='gelu',
                 output_attention=False, distil=True,
                 device=torch.device('cuda:0')):
        '''
        :param conv_archs: cnn 网络结构
        :param cnn_input_dim:  cnn 分支输入维度
        :param enc_in: 编码器输入大小, Informer 输入数据维度
        :param c_out:  分类数
        :param factor: Probesparse attn因子（默认为5）
        :param d_model: 模型维度
        :param n_heads: 多头注意力头数
        :param e_layers: 编码器层数 默认 2 层
        :param d_ff: 模型中全连接网络（FCN）的维度，默认值为512
        :param dropout: dropout概率
        :param attn: 编码器中使用的注意事项（默认为prob）。这可以设置为prob（informer）、full（transformer） ，默认为"prob"论文的主要改进点，提出的注意力机制
        :param embed: 时间特征编码（默认为timeF）。这可以设置为时间F、固定、学习
        :param activation: 激活函数（默认为gelu）
        :param output_attention:
        :param distil:
        :param mix:
        :param device:
        '''
        # 调用父类的初始化方法
        super(CNNInformerMATTModel, self).__init__()

        # cnn 1D 卷积池化网络结构
        self.conv_archs = conv_archs  # cnn 1D 卷积池化网络结构
        self.input_channels = cnn_input_dim  # 分支 CNN输入通道数
        self.cnn1d_features = self.make_1dcnn_layers()  # 1D 卷积池化

        # 注意力机制的选择
        self.attn = attn
        # 是否输出注意力权重  默认不输出
        self.output_attention = output_attention

        # Encoding  # 编码器嵌入层
        self.enc_embedding = DataEmbedding(enc_in, d_model, dropout)

        # Attention  # 选择注意力机制
        Attn = ProbAttention if attn == 'prob' else FullAttention

        # Encoder  # 初始化编码器
        self.encoder = Encoder(
            [
                EncoderLayer(
                    AttentionLayer(Attn(False, factor, attention_dropout=dropout, output_attention=output_attention),
                                   d_model, n_heads, mix=False),
                    d_model,
                    d_ff,
                    dropout=dropout,
                    activation=activation
                ) for l in range(e_layers)  # 根据e_layers参数构建多个编码器层
            ],
            [
                ConvLayer(
                    d_model
                ) for l in range(e_layers - 1)
            ] if distil else None,
            # 层归一化
            norm_layer=torch.nn.LayerNorm(d_model)
        )

        # 多头注意力融合
        self.multiattn_fusion = MultiHeadedAttention(4, d_model)

        # 平均池化
        self.avgpool1d = nn.AdaptiveAvgPool1d(1)

        # 定义全连接层
        # d_model=128   c_out: 10
        self.classifier = nn.Linear(d_model, c_out)

    # CNN-1D 卷积池化结构
    def make_1dcnn_layers(self):
        layers = []
        for (num_convs, out_channels) in self.conv_archs:
            for _ in range(num_convs):
                layers.append(nn.Conv1d(self.input_channels, out_channels, kernel_size=3, padding=1))
                layers.append(nn.ReLU(inplace=True))
                self.input_channels = out_channels
            layers.append(nn.MaxPool1d(kernel_size=2, stride=2))  # 添加池化层
        return nn.Sequential(*layers)

    def forward(self, input_seq):
        # print(input_seq.size())
        # input_seq :  torch.Size([32, 5, 1024])
        batch_size = input_seq.size(0)

        # 时域特征: 送入Informer
        # 数据预处理                                                      5* 16 * 64
        # 注意：这里是 把数据进行了堆叠 把一个5*1024 的矩阵 进行 划分堆叠成形状为 (5*8) * 128 就使输入序列的长度降下来了
        x_enc = input_seq.view(batch_size, 128, 40)  # torch.Size([32, 128, 40])
        # 当然， 还可以 堆叠 为其他形状的矩阵
        # print(x_enc.size()) # torch.Size([32, 128, 40])
        # 编码器部分
        # 输入数据嵌入
        enc_out = self.enc_embedding(x_enc)
        # print(enc_out.size())  # torch.Size([32, 128, 128])
        # 编码器输出和注意力权重
        enc_out, attns = self.encoder(enc_out)
        # print(enc_out.size())  # torch.Size([32, 64, 128])

        # 分支二：1D - CNN
        # 空间特征 卷积池化处理
        #  输入 （batch_size, channels, length）
        cnn1d_features = self.cnn1d_features(input_seq)
        # print(cnn1d_features.size())  # torch.Size([32, 128, 64])
        cnn1d_features = cnn1d_features.permute(0, 2, 1)  # torch.Size([32, 64, 128])

        # 多头注意力融合
        query = enc_out + cnn1d_features  # 要保证特征矩阵形状一样！！！
        # query = query.to(device)
        combined_features = self.multiattn_fusion(query, query, query)  # torch.Size([32, 64, 128])

        # 自适应平均池化
        combined_features = self.avgpool1d(combined_features.permute(0, 2, 1))  # # torch.Size([32, 128, 1])
        # 平铺
        combined_features = combined_features.view((batch_size, -1))  # torch.Size([32, 128]
        outputs = self.classifier(combined_features)  # torch.Size([32, 10]

        return outputs