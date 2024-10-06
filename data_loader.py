import torch
from torch_geometric.data import Dataset, Data
from rdkit import Chem
from rdkit.Chem import AllChem
import pandas as pd
import numpy as np
import warnings
import warnings
from rdkit import RDLogger
import torch
from rdkit import Chem
from torch_geometric.data import Data

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

'''
def one_hot_encoding(value, possible_values):
    """将值进行独热编码。"""
    encoding = [0] * len(possible_values)
    if value in possible_values:
        idx = possible_values.index(value)
        encoding[idx] = 1
    return encoding

def smiles_to_data(smiles, y):
    mol = Chem.MolFromSmiles(smiles)
    if mol is None:
        return None

    num_atoms = mol.GetNumAtoms()
    x = torch.zeros((num_atoms, 256), dtype=torch.float)  # 初始化所有节点特征为 1024 维的全零矩阵

    possible_atom_types = list(range(1, 119))  # 原子序数范围
    possible_chirality = [
        Chem.rdchem.ChiralType.CHI_UNSPECIFIED,
        Chem.rdchem.ChiralType.CHI_TETRAHEDRAL_CW,
        Chem.rdchem.ChiralType.CHI_TETRAHEDRAL_CCW,
        Chem.rdchem.ChiralType.CHI_OTHER,
    ]
    possible_hybridizations = [
        Chem.rdchem.HybridizationType.SP,
        Chem.rdchem.HybridizationType.SP2,
        Chem.rdchem.HybridizationType.SP3,
        Chem.rdchem.HybridizationType.SP3D,
        Chem.rdchem.HybridizationType.SP3D2,
        Chem.rdchem.HybridizationType.UNSPECIFIED,
    ]
    possible_bond_types = [
        Chem.rdchem.BondType.SINGLE,
        Chem.rdchem.BondType.DOUBLE,
        Chem.rdchem.BondType.TRIPLE,
        Chem.rdchem.BondType.AROMATIC,
    ]

    for i, atom in enumerate(mol.GetAtoms()):
        features = []

        # 当前原子的原子序数（与之前代码一致）
        features.append(atom.GetAtomicNum())

        # 添加原子的手性信息（独热编码）
        chirality = atom.GetChiralTag()
        chirality_one_hot = one_hot_encoding(chirality, possible_chirality)
        features.extend(chirality_one_hot)

        # 原子的度（与多少个原子相连）
        features.append(atom.GetDegree())

        # 形式电荷
        features.append(atom.GetFormalCharge())

        # 总的氢原子数
        features.append(atom.GetTotalNumHs())

        # 杂化轨道类型（独热编码）
        hybridization = atom.GetHybridization()
        hybridization_one_hot = one_hot_encoding(hybridization, possible_hybridizations)
        features.extend(hybridization_one_hot)

        # 芳香性
        features.append(int(atom.GetIsAromatic()))

        # 获取与当前原子相连的键的信息
        bonds = atom.GetBonds()
        bond_features = []
        for bond in bonds:
            bond_type = bond.GetBondType()
            bond_type_one_hot = one_hot_encoding(bond_type, possible_bond_types)
            bond_features.extend(bond_type_one_hot)
            # 由于可能有多个键，这里只取前两个键的特征，超过的忽略，不足的填充0
            if len(bond_features) >= 8:  # 每个键类型独热编码长度为4，两个键共8个特征
                break
        bond_features = bond_features[:8]  # 只保留前8个特征
        bond_features += [0] * (8 - len(bond_features))  # 不足8个特征用0填充
        features.extend(bond_features)

        # 获取邻近的原子信息（与之前代码一致，最多取前三个邻居的原子序数）
        neighbors = [neighbor.GetAtomicNum() for neighbor in atom.GetNeighbors()]
        for j in range(3):  # 确保有三个邻居的位置
            if j < len(neighbors):
                features.append(neighbors[j])
            else:
                features.append(0)  # 如果邻居不足三个，用0填充

        # 将特征列表转换为Tensor，并填充到x的对应行
        feature_length = len(features)
        x[i, :feature_length] = torch.tensor(features, dtype=torch.float)

    # 获取分子中的化学键信息（即边）
    bonds = [(bond.GetBeginAtomIdx(), bond.GetEndAtomIdx()) for bond in mol.GetBonds()]

    # 创建边索引矩阵
    if len(bonds) > 0:
        edge_index = torch.tensor(bonds, dtype=torch.long).t().contiguous()  # 形状为 [2, num_edges]
    else:
        # 如果没有边，添加自环
        edge_index = torch.tensor([[0], [0]], dtype=torch.long)

    # 标签
    y = torch.tensor([y], dtype=torch.float)

    return Data(x=x, edge_index=edge_index, y=y)
'''


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