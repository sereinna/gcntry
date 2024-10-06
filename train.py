import torch
import torch.optim as optim
from losses import PearsonRMSELoss, PearsonRLoss
from utils import log_to_file


def train_model(model, train_loader, val_loader, model_type, seed, log_file, loss_type, train_params, task_type):
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    
    model = model.to(device)
    
    if task_type == 'regression':
        if loss_type == 'PearsonRMSELoss':
            criterion = PearsonRMSELoss()
        elif loss_type == 'PearsonRLoss':
            criterion = PearsonRLoss()
        else:
            criterion = torch.nn.MSELoss()
    else:  # classification
        criterion = torch.nn.BCEWithLogitsLoss()
    
    optimizer = optim.Adam(model.parameters(), lr=train_params['lr'])
    
    for epoch in range(train_params['num_epochs']):
        train_loss = train(model, train_loader, optimizer, criterion, device, task_type)
        val_loss = evaluate(model, val_loader, criterion, device, task_type)
        log_to_file(log_file, f"Epoch {epoch+1}/{train_params['num_epochs']}, Train Loss: {train_loss:.4f}, Val Loss: {val_loss:.4f}")
    
    return model

def train(model, loader, optimizer, criterion, device, task_type):
    model.train()
    total_loss = 0
    for data in loader:
        data = data.to(device)
        optimizer.zero_grad()
        output = model(data)
        if task_type == 'classification':
            loss = criterion(output, data.y.float())
        else:
            loss = criterion(output, data.y)
        loss.backward()
        optimizer.step()
        total_loss += loss.item()
    return total_loss / len(loader)

def evaluate(model, loader, criterion, device, task_type):
    model.eval()
    total_loss = 0
    with torch.no_grad():
        for data in loader:
            data = data.to(device)
            output = model(data)
            if task_type == 'classification':
                loss = criterion(output, data.y.float())
            else:
                loss = criterion(output, data.y)
            total_loss += loss.item()
    return total_loss / len(loader)