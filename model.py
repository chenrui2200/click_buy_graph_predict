from torch_geometric.nn import global_mean_pool as gap
import torch
import torch.nn.functional as F
from torch_geometric.nn import SAGEConv, TopKPooling, global_add_pool as gap

embed_dim = 128
root_dir='D:\\github_training\\youchoosebuy\\data'

class GraphNet(torch.nn.Module):
    """Graph Neural Network for binary classification tasks."""

    def __init__(self, emb_dim):
        super(GraphNet, self).__init__()
        # 定义图卷积层
        self.conv1 = SAGEConv(emb_dim, 128)
        self.pool1 = TopKPooling(128, ratio=0.8)

        self.conv2 = SAGEConv(128, 128)
        self.pool2 = TopKPooling(128, ratio=0.8)

        self.conv3 = SAGEConv(128, 128)
        self.pool3 = TopKPooling(128, ratio=0.8)

        # 定义嵌入层
        self.item_embedding = torch.nn.Embedding(num_embeddings=emb_dim + 10, embedding_dim=emb_dim)

        # 定义线性层
        self.lin1 = torch.nn.Linear(128, 128)
        self.lin2 = torch.nn.Linear(128, 64)
        self.lin3 = torch.nn.Linear(64, 1)

        # 定义标准化层和激活函数
        self.bn1 = torch.nn.BatchNorm1d(128)
        self.bn2 = torch.nn.BatchNorm1d(64)
        self.act1 = torch.nn.ReLU()
        self.act2 = torch.nn.ReLU()

    def forward(self, data):
        x, edge_index, batch = data.x, data.edge_index, data.batch

        # 使用嵌入层将节点特征进行编码
        x = self.item_embedding(x).squeeze(1)  # n * 128

        # 第一个卷积层和池化
        x = F.relu(self.conv1(x, edge_index))
        x, edge_index, _, batch, _, _ = self.pool1(x, edge_index, None, batch)
        x1 = gap(x, batch)  # 全局池化

        # 第二个卷积层和池化
        x = F.relu(self.conv2(x, edge_index))
        x, edge_index, _, batch, _, _ = self.pool2(x, edge_index, None, batch)
        x2 = gap(x, batch)  # 全局池化

        # 第三个卷积层和池化
        x = F.relu(self.conv3(x, edge_index))
        x, edge_index, _, batch, _, _ = self.pool3(x, edge_index, None, batch)
        x3 = gap(x, batch)  # 全局池化

        # 将三个尺度的全局特征相加
        x = x1 + x2 + x3

        # 通过线性层进行特征转换
        x = self.act1(self.lin1(x))
        x = self.act2(self.lin2(x))
        x = F.dropout(x, p=0.5, training=self.training)

        # 最后一层使用Sigmoid激活函数进行二元分类
        x = torch.sigmoid(self.lin3(x)).squeeze(1)  # batch个结果
        return x

