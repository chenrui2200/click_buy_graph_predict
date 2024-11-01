# 加载模型
import numpy as np
import pandas as pd
from sklearn.preprocessing import LabelEncoder

from model import GraphNet
from torch_geometric.loader import DataLoader
from model import GraphNet, root_dir
from torch_geometric.data import InMemoryDataset
from tqdm import tqdm
import torch
from torch_geometric.data import Data

if torch.cuda.is_available():
    device = "cuda"
else:
    device = "cpu"

loaded_model = GraphNet(emb_dim=100).to(device)
loaded_model.load_state_dict(torch.load('graphnet_model.pth'))
loaded_model.eval()  # 设置为评估模式

# 准备预测数据（假设你有新的点击数据）
# 这里你可以使用与训练时相同的处理方式来创建新的 Data 对象
# 例如，假设你有新的点击数据文件
new_clicks_df = pd.read_csv('test-yoochoose-clicks.dat', sep=';', names=['session_id', 'timestamp', 'item_id', 'category'])

# 创建新的图数据
new_data_list = []
for session_id, group in tqdm(new_clicks_df.groupby('session_id'), desc="Processing new sessions"):
    sess_item_id = LabelEncoder().fit_transform(group.item_id)
    group = group.reset_index(drop=True)
    group['sess_item_id'] = sess_item_id

    # 节点特征
    node_features = group['sess_item_id'].values
    node_features = torch.LongTensor(node_features).unsqueeze(1)

    # 创建边缘索引
    if len(group) > 1:  # 确保有足够的节点以构建边
        target_nodes = group.sess_item_id.values[1:]
        source_nodes = group.sess_item_id.values[:-1]
        edge_index = torch.tensor([source_nodes, target_nodes], dtype=torch.long)

        # 创建图数据对象
        x = node_features
        new_data = Data(x=x, edge_index=edge_index)

        new_data_list.append(new_data)

# 将新的图数据转换为 DataLoader
new_loader = DataLoader(new_data_list, batch_size=32, shuffle=False)

# 开始预测
predictions = []
with torch.no_grad():  # 禁用梯度计算以节省内存
    for data in new_loader:
        data = data.to(device)
        output = loaded_model(data)
        predictions.append(output.cpu().numpy())

# 将预测结果合并
predictions = np.concatenate(predictions)

# 打印预测结果
print("Predictions:", predictions)