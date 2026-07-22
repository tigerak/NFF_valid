import torch
import torch.nn.functional as F

def mixup(x, y, alpha=1.0):
    """Mixup: 두 샘플을 선형 보간"""
    batch_size = x.size(0)
    index = torch.randperm(batch_size)
    
    lam = torch.distributions.Beta(alpha, alpha).sample().item()
    
    mixed_x = lam * x + (1 - lam) * x[index]
    y_a, y_b = y, y[index]
    
    return mixed_x, y_a, y_b, lam

def cutmix(x, y, alpha=1.0):
    """Cutmix: 이미지 영역을 교환"""
    batch_size = x.size(0)
    index = torch.randperm(batch_size)
    
    lam = torch.distributions.Beta(alpha, alpha).sample().item()
    
    # 무작위 박스 생성
    _, _, h, w = x.size()
    cut_ratio = (1 - lam) ** 0.5
    cut_h = int(h * cut_ratio)
    cut_w = int(w * cut_ratio)
    
    cx = torch.randint(0, w, (1,)).item()
    cy = torch.randint(0, h, (1,)).item()
    
    bbx1 = max(0, cx - cut_w // 2)
    bby1 = max(0, cy - cut_h // 2)
    bbx2 = min(w, cx + cut_w // 2)
    bby2 = min(h, cy + cut_h // 2)
    
    mixed_x = x.clone()
    mixed_x[:, :, bby1:bby2, bbx1:bbx2] = x[index, :, bby1:bby2, bbx1:bbx2]
    
    # Lambda 재계산 (실제 영역 비율)
    lam = 1 - (bbx2 - bbx1) * (bby2 - bby1) / (h * w)
    
    y_a, y_b = y, y[index]
    
    return mixed_x, y_a, y_b, lam