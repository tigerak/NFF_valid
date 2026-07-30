import torch 
import torch.nn as nn
import torch.nn.functional as F
from torch.nn.functional import cross_entropy
import math



def criterion(outputs, targets):
    return nn.CrossEntropyLoss()(outputs, targets)


## class 별로 
def focal_loss(logits, targets, alpha=0.25, gamma=2.0, reduction='mean'):
    ce_loss = F.cross_entropy(logits, targets, reduction='none')
    pt = torch.exp(-ce_loss)  # 확률 변환
    
    if alpha is None:
        alpha_factor = torch.ones_like(ce_loss)
    
    elif isinstance(alpha, (list, tuple)):
        alpha_t = torch.tensor(alpha, dtype=logits.dtype, device=logits.device)
        alpha_factor = alpha_t[targets]
    elif torch.is_tensor(alpha):
        alpha_t = alpha.to(device=logits.device, dtype=logits.dtype)
        if alpha_t.ndim == 0:
            alpha_factor = torch.full_like(ce_loss, float(alpha_t.item()))
        else:
            alpha_factor = alpha_t[targets]
    else:
        alpha_factor = torch.full_like(ce_loss, float(alpha))

    loss = alpha_factor * (1.0 - pt).pow(gamma) * ce_loss

    if reduction == 'sum':
        return loss.sum()
    if reduction == 'none':
        return loss
    return loss.mean()

def loss_dynamic(outputs, targets, special_class=4, special_weight=2.0, current_epoch=0, total_epochs=100):
    base_loss = nn.CrossEntropyLoss()(outputs, targets)
    
    # special_class가 int 인 경우와 list/tuple 인 경우를 구분
    if isinstance(special_class, int):
        special_mask = (targets == special_class)
    else:
    
        special_mask = torch.isin(targets, torch.tensor(special_class).to(targets.device))
        
        # 또는 for loop를 사용하여 mask 생성
        # special_mask = torch.zeros_like(targets, dtype=torch.bool)
        # for sc in special_class:
        #     special_mask |= (targets == sc)
    
    if special_mask.sum() > 0:
        spec_loss = nn.CrossEntropyLoss()(outputs[special_mask], targets[special_mask])
        epoch_factor = current_epoch / total_epochs
        special_weight = special_weight * epoch_factor
        base_loss += special_weight * spec_loss

    return base_loss

def calculate_loss_per_class(class_losses, loss_fn, logits, labels):

    """각 클래스별로 손실값을 계산"""
    batch_size = labels.shape[0]
    
    for i in range(batch_size):
        class_idx = labels[i].item()  # 현재 샘플의 클래스 인덱스
        loss = loss_fn(logits[i].unsqueeze(0), labels[i].unsqueeze(0))
        class_losses[class_idx].append(loss.item()) 
    
    for class_idx, losses in class_losses.items():
        if len(losses) > 0:
            avg_loss = sum(losses) / len(losses)
        else:
            avg_loss = 0.0 


class SubCenterArcFace(nn.Module):
    """Sub-center ArcFace loss.

    각 클래스마다 여러 sub-center를 두고, 클래스 점수는 그중 가장 높은 center로 계산한다.
    pruning은 warmup 이후에만 수행하며, 한 번에 클래스당 최대 1개만 비활성화한다.
    """

    def __init__(self, num_classes, feature_dim, num_sub_centers=4, margin=0.3, scale=64.0):
        super().__init__()
        self.num_classes = int(num_classes)
        self.feature_dim = int(feature_dim)
        self.num_sub_centers = int(num_sub_centers)
        self.margin = float(margin)
        self.scale = float(scale)

        self.weight = nn.Parameter(torch.empty(self.num_classes, self.num_sub_centers, self.feature_dim))
        nn.init.xavier_uniform_(self.weight)

        self.register_buffer('active_mask', torch.ones(self.num_classes, self.num_sub_centers, dtype=torch.bool))
        self.register_buffer('sub_center_usage', torch.zeros(self.num_classes, self.num_sub_centers))

    def _cosine_logits(self, features):
        # Feature / center를 모두 L2 normalize해서 cosine similarity를 직접 로짓으로 사용한다.
        features = F.normalize(features, dim=1)
        weight = F.normalize(self.weight, dim=2)
        return torch.einsum('bd,ckd->bck', features, weight)

    def inference_logits(self, features):
        """Margin 없이 ArcFace 분류 logits를 반환합니다.

        검증/평가에서는 ground-truth label 기반 margin 조작을 하지 않고,
        각 클래스에서 가장 유사한 sub-center의 cosine 값(max over K)을
        scale만 곱한 logits로 사용합니다.
        """
        cosine = self._cosine_logits(features)
        cosine = cosine.masked_fill(~self.active_mask.unsqueeze(0), -1e4)
        class_logits, _ = cosine.max(dim=2)
        return class_logits * self.scale

    def forward(self, features, labels):
        batch_size = features.size(0)
        device = features.device

        cosine = self._cosine_logits(features)
        cosine = cosine.masked_fill(~self.active_mask.unsqueeze(0), -1e4)

        class_logits, class_best_idx = cosine.max(dim=2)

        batch_indices = torch.arange(batch_size, device=device)
        target_cos = class_logits[batch_indices, labels].clamp(-1.0 + 1e-7, 1.0 - 1e-7)
        theta = torch.acos(target_cos)
        class_logits[batch_indices, labels] = torch.cos(theta + self.margin)

        target_center_idx = class_best_idx[batch_indices, labels]
        self.sub_center_usage[labels, target_center_idx] += 1

        return F.cross_entropy(class_logits * self.scale, labels)

    def prune_unused_subcenters(self, warmup_epochs=3, current_epoch=0, min_usage=1, max_remove_per_class=1):
        """사용량이 낮은 sub-center를 보수적으로 비활성화한다.

        - warmup 이전에는 pruning을 수행하지 않고 usage 통계만 초기화
        - 클래스당 최소 1개 center는 반드시 유지
        - 한 번의 호출에서 클래스당 최대 max_remove_per_class만 제거
        """
        if current_epoch < warmup_epochs:
            self.sub_center_usage.zero_()
            return 0

        removed_total = 0
        with torch.no_grad():
            for class_idx in range(self.num_classes):
                active_idx = torch.nonzero(self.active_mask[class_idx], as_tuple=False).flatten()
                if active_idx.numel() <= 1:
                    continue

                usage = self.sub_center_usage[class_idx, active_idx]
                low_usage_order = torch.argsort(usage)

                removed_for_class = 0
                for order_idx in low_usage_order.tolist():
                    center_idx = active_idx[order_idx].item()
                    if usage[order_idx].item() > min_usage:
                        break
                    if self.active_mask[class_idx].sum().item() <= 1:
                        break
                    self.active_mask[class_idx, center_idx] = False
                    removed_total += 1
                    removed_for_class += 1
                    if removed_for_class >= max_remove_per_class:
                        break

            # 다음 pruning 주기 전까지 usage를 다시 누적하기 위해 초기화.
            self.sub_center_usage.zero_()

        return removed_total


def extract_model_state_dict(weight_obj):
    """체크포인트 호환 로더.

    새 포맷(dict: model_state_dict/arcface_state_dict/...)과
    구 포맷(raw state_dict) 둘 다 지원한다.
    """
    if isinstance(weight_obj, dict) and 'model_state_dict' in weight_obj:
        return weight_obj['model_state_dict']
    return weight_obj


class SupConLoss(nn.Module):
    def __init__(self, temperature=0.07):
        super().__init__()
        self.temperature = temperature

    def forward(self, features, labels):

        device = features.device
        B = labels.shape[0]
        if features.shape[0] != 2 * B:
            raise ValueError("features should be 2*B in first dim when labels length is B")

        # normalize already done upstream, but ensure:
        features = F.normalize(features, dim=1)

        # create labels for 2*B: duplicate labels for each view
        labels = labels.repeat(2)  # [2*B]

        # similarity matrix
        sim = torch.div(torch.matmul(features, features.T), self.temperature)  # [2B,2B]

        # mask to remove self-comparisons
        mask = torch.eye(2*B, dtype=torch.bool, device=device)
        sim_masked = sim.masked_fill(mask, -1e9)  # or large negative

        # create positive mask: positive if same label and not same index
        labels_eq = labels.unsqueeze(0) == labels.unsqueeze(1)  # [2B,2B]
        positives_mask = labels_eq & ~mask  # exclude self

        # For each anchor i, compute log_prob of positives among all non-self
        exp_sim = torch.exp(sim_masked)
        exp_sim_sum = exp_sim.sum(dim=1, keepdim=True)  # denom

        # sum of positives exp
        pos_exp_sum = (exp_sim * positives_mask.float()).sum(dim=1)

        # avoid zero positives (happens if class appears only once in batch) -> exclude those anchors from loss
        non_zero_pos = pos_exp_sum > 0
        # compute loss only for anchors that have positives
        loss = -torch.log( (pos_exp_sum[non_zero_pos] / exp_sim_sum[non_zero_pos].squeeze(1)) + 1e-12 )
        if loss.numel() == 0:
            return torch.tensor(0.0, device=device, requires_grad=True)
        return loss.mean()
