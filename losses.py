import torch
import torch.nn as nn
import torch.nn.functional as F

class PearsonRMSELoss(nn.Module):
    def __init__(self, epsilon=1e-8):
        super(PearsonRMSELoss, self).__init__()
        self.epsilon = epsilon

    def forward(self, output, target):
        mse = F.mse_loss(output, target)
        pearson = self.pearson_r(output, target)
        return mse + (1 - pearson)

    def pearson_r(self, x, y):
        vx = x - torch.mean(x)
        vy = y - torch.mean(y)
        return torch.sum(vx * vy) / (torch.sqrt(torch.sum(vx ** 2) + self.epsilon) * torch.sqrt(torch.sum(vy ** 2) + self.epsilon))

class PearsonRLoss(nn.Module):
    def __init__(self, epsilon=1e-8):
        super(PearsonRLoss, self).__init__()
        self.epsilon = epsilon

    def forward(self, output, target):
        r = self.pearson_r(output, target)
        return 1 - r

    def pearson_r(self, output, target):
        mean_x = torch.mean(output)
        mean_y = torch.mean(target)
        xm = output - mean_x
        ym = target - mean_y
        r_num = torch.sum(xm * ym)
        r_den = torch.sqrt(torch.sum(xm.pow(2)) + self.epsilon) * torch.sqrt(torch.sum(ym.pow(2)) + self.epsilon)
        r = r_num / r_den
        return torch.clamp(r, -1.0, 1.0)