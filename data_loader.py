import torch
from torch_geometric.data import Dataset, Data
from rdkit import Chem
from rdkit.Chem import AllChem
import pandas as pd
import numpy as np
import warnings
import warnings
from rdkit import RDLogger

# 禁用 RDKit 的日志
RDLogger.DisableLog('rdApp.*')

# 忽略特定的弃用警告
warnings.filterwarnings("ignore", category=DeprecationWarning, message="please use MorganGenerator")


def smiles_to_data(smiles, y):
    mol = Chem.MolFromSmiles(smiles)
    if mol is None:
        return None
    
    # 使用 GetMorganFingerprintAsBitVect 计算 Morgan 指纹
    fp = AllChem.GetMorganFingerprintAsBitVect(mol, 2, nBits=1024)
    fp_array = np.array(fp)
    
    # 使用 Morgan 指纹作为单个节点特征
    x = torch.tensor(fp_array, dtype=torch.float).unsqueeze(0)  # 形状为 [1, 1024]
    
    # 创建只有一个节点的图
    edge_index = torch.tensor([[0], [0]], dtype=torch.long)  # 自环
    
    y = torch.tensor([y], dtype=torch.float)
    
    return Data(x=x, edge_index=edge_index, y=y)

class MoleculeDataset(Dataset):
    def __init__(self, csv_file, target_column):
        self.data_list = []
        data_frame = pd.read_csv(csv_file)
        for _, row in data_frame.iterrows():
            mol = Chem.MolFromSmiles(row['smiles'])
            if mol is not None:
                data = smiles_to_data(row['smiles'], row[target_column])
                if data is not None:
                    self.data_list.append(data)
    
    def __len__(self):
        return len(self.data_list)
    
    def __getitem__(self, idx):
        return self.data_list[idx]

def load_dataset(name, optimized=False):
    if optimized:
        if name == 'bbbp':
            return MoleculeDataset('data/BBBP_bupingheng.csv', 'p_np')
        elif name == 'esol':
            return MoleculeDataset('data/delaney-processed_bupingheng.csv', 'ESOL predicted log solubility in mols per litre')
        elif name == 'lipo':
            return MoleculeDataset('data/Lipophilicity_bupingheng.csv', 'exp')
    else:
        if name == 'bbbp':
            return MoleculeDataset('data/BBBP.csv', 'p_np')
        elif name == 'esol':
            return MoleculeDataset('data/delaney-processed.csv', 'ESOL predicted log solubility in mols per litre')
        elif name == 'lipo':
            return MoleculeDataset('data/Lipophilicity.csv', 'exp')
    
    raise ValueError("不支持的数据集名称")