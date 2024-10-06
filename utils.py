import torch
import random

# 设置随机种子
def set_seed(seed):
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    random.seed(seed)

# 保存日志到文本文件
def log_to_file(log_file, message):
    with open(log_file, 'a') as f:
        print(message)
        f.write(message + '\n')
