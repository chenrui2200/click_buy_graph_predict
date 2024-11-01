import numpy as np
from sklearn.metrics import roc_auc_score
from torch_geometric.loader import DataLoader
from model import GraphNet, root_dir
from torch_geometric.data import InMemoryDataset
from tqdm import tqdm
import torch
from torch_geometric.data import Data
from sklearn.preprocessing import LabelEncoder
import pandas as pd


if torch.cuda.is_available():
    device = "cuda"
else:
    device = "cpu"


import pandas as pd
import torch
from torch_geometric.data import InMemoryDataset, Data
from sklearn.preprocessing import LabelEncoder
from tqdm import tqdm

class ChooseBinaryDataset(InMemoryDataset):
    def __init__(self, root, clicks_file, buys_file, transform=None, pre_transform=None):
        self.clicks_file = clicks_file
        self.buys_file = buys_file
        super(ChooseBinaryDataset, self).__init__(root, transform, pre_transform)
        self.data, self.slices = self.process()

    @property
    def processed_file_names(self):
        return ['data.pt']  # 指定处理后保存的数据文件名

    def process(self):
        # 加载数据
        clicks_df = pd.read_csv(self.clicks_file, sep=';', names=['session_id', 'timestamp', 'item_id', 'category'])
        buys_df = pd.read_csv(self.buys_file, sep=';', names=['session_id', 'timestamp', 'item_id', 'price', 'quantity'])

        # 创建购买标签
        buys_df['buy'] = 1  # 标记购买的商品
        clicks_df = clicks_df.merge(buys_df[['session_id', 'item_id', 'buy']], on=['session_id', 'item_id'], how='left')
        clicks_df['buy'] = clicks_df['buy'].fillna(0)  # 填充未购买的商品为0

        # 存储图数据对象
        data_list = []
        grouped = clicks_df.groupby('session_id')

        for session_id, group in tqdm(grouped, desc="Processing sessions"):
            # 标签编码
            sess_item_id = LabelEncoder().fit_transform(group.item_id)
            group = group.reset_index(drop=True)
            group['sess_item_id'] = sess_item_id

            # 节点特征
            node_features = group['sess_item_id'].values
            node_features = torch.LongTensor(node_features).unsqueeze(1)

            # 创建边缘索引
            target_nodes = group.sess_item_id.values[1:]
            source_nodes = group.sess_item_id.values[:-1]
            edge_index = torch.tensor([source_nodes, target_nodes], dtype=torch.long)

            # 创建图数据对象
            x = node_features
            y = torch.FloatTensor([group.buy.values[0]])  # 使用第一个商品的购买标签

            data = Data(x=x, edge_index=edge_index, y=y)
            data_list.append(data)

        data, slices = self.collate(data_list)
        torch.save((data, slices), self.processed_paths[0])
        return data, slices


# 使用示例
dataset = ChooseBinaryDataset(root=root_dir,
                              clicks_file='yoochoose-clicks.dat',
                              buys_file='yoochoose-buys.dat')

model = GraphNet(emb_dim=100).to(device)  # 使用适当的嵌入维度
optimizer = torch.optim.Adam(model.parameters(), lr=0.001)
criterion = torch.nn.BCELoss()

# 创建数据加载器
train_loader = DataLoader(dataset, batch_size=32, shuffle=True)

# 训练循环
for epoch in range(100):  # 设置适当的 epoch 数
    model.train()
    total_loss = 0

    for data in train_loader:
        data = data.to(device)
        optimizer.zero_grad()
        output = model(data)
        loss = criterion(output, data.y)
        loss.backward()
        optimizer.step()
        total_loss += loss.item()

    print(f'Epoch {epoch + 1}, Loss: {total_loss / len(train_loader)}')

torch.save(model.state_dict(), 'graphnet_model.pth')
print("Model saved to graphnet_model.pth")
