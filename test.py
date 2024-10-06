import torch
from utils import log_to_file
from losses import PearsonRMSELoss, PearsonRLoss
import torch.nn.functional as F
from sklearn.metrics import r2_score, roc_auc_score, average_precision_score, accuracy_score, precision_recall_curve, f1_score
from sklearn.metrics import auc as sk_auc

def compute_regression_metrics(output, target):
    mae = F.l1_loss(output, target, reduction='mean').item()
    mse = F.mse_loss(output, target, reduction='mean').item()
    rmse = torch.sqrt(torch.tensor(mse)).item()
    r2 = r2_score(target.cpu().numpy(), output.cpu().detach().numpy())
    return mae, mse, rmse, r2

def compute_classification_metrics(output, target):
    output_prob = torch.sigmoid(output)
    output_class = (output_prob > 0.5).float()
    
    auc = roc_auc_score(target.cpu().numpy(), output_prob.cpu().detach().numpy())
    prauc = average_precision_score(target.cpu().numpy(), output_prob.cpu().detach().numpy())
    acc = accuracy_score(target.cpu().numpy(), output_class.cpu().detach().numpy())
    f1 = f1_score(target.cpu().numpy(), output_class.cpu().detach().numpy())
    
    precision, recall, _ = precision_recall_curve(target.cpu().numpy(), output_prob.cpu().detach().numpy())
    prc = sk_auc(recall, precision)  # 使用 sk_auc 而不是 auc
    
    return auc, prauc, acc, prc, f1

def evaluate_model(model, train_loader, val_loader, test_loader, model_type, log_file, loss_type, task_type):
    results = {}
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    
    if loss_type == 'PearsonRMSELoss':
        criterion = PearsonRMSELoss()
    elif loss_type == 'PearsonRLoss':
        criterion = PearsonRLoss()
    elif task_type == 'classification':
        criterion = torch.nn.BCEWithLogitsLoss()
    else:
        criterion = torch.nn.MSELoss()

    model.to(device)
    model.eval()

    for name, loader in [('train', train_loader), ('val', val_loader), ('test', test_loader)]:
        if task_type == 'regression':
            metrics = calculate_regression_metrics_over_loader(model, loader, criterion, device)
            log_to_file(log_file, f"{name.capitalize()} Pearson R: {metrics['pearson_r']:.4f}, MAE: {metrics['mae']:.4f}, MSE: {metrics['mse']:.4f}, RMSE: {metrics['rmse']:.4f}, R2: {metrics['r2']:.4f}")
        else:
            metrics = calculate_classification_metrics_over_loader(model, loader, criterion, device)
            log_to_file(log_file, f"{name.capitalize()} AUC: {metrics['auc']:.4f}, PRAUC: {metrics['prauc']:.4f}, ACC: {metrics['acc']:.4f}, PRC: {metrics['prc']:.4f}, F1: {metrics['f1']:.4f}")
        
        results[name] = metrics

    return {
        'model': model_type,
        'train_metrics': results['train'],
        'val_metrics': results['val'],
        'test_metrics': results['test']
    }

def calculate_regression_metrics_over_loader(model, loader, criterion, device):
    total_pearson_r = 0
    total_mae, total_mse, total_rmse, total_r2 = 0, 0, 0, 0
    
    with torch.no_grad():
        for data in loader:
            data = data.to(device)
            output = model(data)
            target = data.y

            pearson_r = criterion.pearson_r(output.flatten(), target.flatten())
            total_pearson_r += pearson_r

            mae, mse, rmse, r2 = compute_regression_metrics(output.flatten(), target.flatten())
            total_mae += mae
            total_mse += mse
            total_rmse += rmse
            total_r2 += r2

    num_batches = len(loader)
    return {
        'pearson_r': total_pearson_r / num_batches,
        'mae': total_mae / num_batches,
        'mse': total_mse / num_batches,
        'rmse': total_rmse / num_batches,
        'r2': total_r2 / num_batches
    }

def calculate_classification_metrics_over_loader(model, loader, criterion, device):
    total_auc, total_prauc, total_acc, total_prc, total_f1 = 0, 0, 0, 0, 0
    
    with torch.no_grad():
        for data in loader:
            data = data.to(device)
            output = model(data)
            target = data.y

            auc, prauc, acc, prc, f1 = compute_classification_metrics(output.flatten(), target.flatten())
            total_auc += auc
            total_prauc += prauc
            total_acc += acc
            total_prc += prc
            total_f1 += f1

    num_batches = len(loader)
    return {
        'auc': total_auc / num_batches,
        'prauc': total_prauc / num_batches,
        'acc': total_acc / num_batches,
        'prc': total_prc / num_batches,
        'f1': total_f1 / num_batches
    }