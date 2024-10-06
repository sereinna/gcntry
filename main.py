from utils import set_seed, log_to_file
from data_loader import load_dataset
from train import train_model
from test import evaluate_model
import os
import torch
import pandas as pd
from datetime import datetime
from sklearn.model_selection import train_test_split
from model import GNNModel
from collections import Counter
from torch.utils.data import random_split
import random
from torch.utils.data import Subset
from torch_geometric.loader import DataLoader



# 设置超参数和其他配置
dataset_name = 'lipo'  # 选择数据集，'bbbp', 'esol', 'lipo' 可供选择
use_optimized = False  # 是否使用优化后的数据集，False为原始数据集，True为优化版本
#models_to_run = ['GCN', 'GAT', 'MPNN']  # 选择多个模型进行训练
models_to_run = ['GCN']  # 选择多个模型进行训练
#seeds = [1, 12, 123, 1234]  # 设置不同的随机种子
seeds = [1]  # 设置不同的随机种子
split_ratio = [0.8, 0.1, 0.1]  # 训练集、验证集和测试集的分割比例，和必须为1
log_folder = 'logs'  # 日志保存目录
loss_type = 'PearsonRMSELoss'  # 损失函数类型，可以选择 'PearsonRMSELoss', 'PearsonRLoss' 或 'MSELoss'
task_type = 'regression' if dataset_name in ['esol', 'lipo'] else 'classification'



def print_class_distribution(dataset, name):
    labels = [data.y.item() for data in dataset]
    distribution = Counter(labels)
    print(f"{name} dataset class distribution:", distribution)


# 模型超参数
model_params = {
    'GCN': {'num_node_features': 1024, 'hidden_dim': 64},
    'GAT': {'num_node_features': 1024, 'hidden_dim': 64},
    'MPNN': {'num_node_features': 1024, 'hidden_dim': 64}
}

# 训练超参数
train_params = {
    'lr': 0.0001,
    'batch_size': 128,
    'num_epochs': 100
}


def split_dataset(dataset, split_ratio, seed):
    assert sum(split_ratio) == 1.0, "Split ratios must sum to 1"
    
    torch.manual_seed(seed)
    
    dataset_size = len(dataset)
    lengths = [int(r * dataset_size) for r in split_ratio]
    lengths[-1] = dataset_size - sum(lengths[:-1])  # 确保总和等于样本总数
    
    return random_split(dataset, lengths, generator=torch.Generator().manual_seed(seed))
# 创建日志目录
if not os.path.exists(log_folder):
    os.makedirs(log_folder)

def run_experiment():
    dataset = load_dataset(dataset_name, use_optimized)

    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    log_file = os.path.join(log_folder, f'{dataset_name}_run_{timestamp}.txt')
    log_to_file(log_file, f"Running on dataset: {dataset_name}, with split ratio (train, val, test): {split_ratio}")

    results = []
    for seed in seeds:
        log_to_file(log_file, f"Running experiment for seed: {seed}")
        set_seed(seed)

        # 获取数据集大小
        dataset_size = len(dataset)
        train_size = int(split_ratio[0] * dataset_size)
        val_size = int(split_ratio[1] * dataset_size)
        test_size = dataset_size - train_size - val_size

        # 生成随机索引并打乱
        indices = list(range(dataset_size))
        random.Random(seed).shuffle(indices)

        # 分割索引
        train_indices = indices[:train_size]
        val_indices = indices[train_size:train_size+val_size]
        test_indices = indices[train_size+val_size:]

        # 创建数据子集
        train_dataset = Subset(dataset, train_indices)
        val_dataset = Subset(dataset, val_indices)
        test_dataset = Subset(dataset, test_indices)

        train_loader = DataLoader(train_dataset, batch_size=train_params['batch_size'], shuffle=True)
        val_loader = DataLoader(val_dataset, batch_size=train_params['batch_size'])
        test_loader = DataLoader(test_dataset, batch_size=train_params['batch_size'])


        #print_class_distribution(dataset, "Full")
        #print_class_distribution(train_dataset, "Train")
        #print_class_distribution(val_dataset, "Validation")
        #print_class_distribution(test_dataset, "Test")


        for model_type in models_to_run:
            model_path = os.path.join( 'output', 'pt', f'{model_type}_seed_{seed}_{dataset_name}_model.pth')
            
            if not os.path.exists(model_path):
                # Train and save model
                model = GNNModel(model_type=model_type, **model_params[model_type])
                model = train_model(model, train_loader, val_loader, model_type, seed, log_file, loss_type, train_params, task_type)
                torch.save(model.state_dict(), model_path)
            else:
                # Load pre-trained model
                model = GNNModel(model_type=model_type, **model_params[model_type])
                model.load_state_dict(torch.load(model_path))

            # Evaluate model
            model_results = evaluate_model(model, train_loader, val_loader, test_loader, model_type, log_file, loss_type, task_type)
            results.append(model_results)
    
    # Save results
    results_df = pd.DataFrame(results)
    model_type = models_to_run
    results_file = os.path.join( 'output', 'csv', f'{model_type}_{dataset_name}_results_{timestamp}.csv')
    results_df.to_csv(results_file, index=False, encoding='utf-8-sig')
    log_to_file(log_file, f"Training results saved to {results_file}")

if __name__ == '__main__':
    run_experiment()
