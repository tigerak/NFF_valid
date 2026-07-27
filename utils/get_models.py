import os 
import torch 
import torch.nn as nn
import torch.nn.functional as F
from peft import LoraConfig, get_peft_model

# from models import timm
import timm

def get_timm_model(args):
    
    model = timm.create_model(args.model_name, pretrained=False)
    weight = torch.load(os.path.join(args.model_dir, args.pretrained), map_location="cuda")
    model.load_state_dict(weight, strict=False)

    return model


def _replace_classifier_head(model, num_classes):
    """Replace backbone classifier head with a num_classes output head.

    Tries timm's reset_classifier first, then falls back to common attributes.
    """
    if hasattr(model, 'reset_classifier'):
        model.reset_classifier(num_classes=num_classes)
        return model

    if hasattr(model, 'head') and hasattr(model.head, 'fc'):
        model.head.fc = nn.Linear(model.head.fc.in_features, num_classes)
        return model

    if hasattr(model, 'classifier') and isinstance(model.classifier, nn.Linear):
        model.classifier = nn.Linear(model.classifier.in_features, num_classes)
        return model

    if hasattr(model, 'fc') and isinstance(model.fc, nn.Linear):
        model.fc = nn.Linear(model.fc.in_features, num_classes)
        return model

    raise ValueError(
        f"Unsupported model head structure for model: {type(model).__name__}. "
        "Please add custom replacement logic in _replace_classifier_head."
    )


def build_model(args):
    """Build a model with configurable classification head strategy.

    Supported head_type values:
    - transformer (default, backward-compatible)
    - linear
    - general_gem
    - effnet_gem
    - backbone (no head replacement)
    """
    head_type = str(getattr(args, 'head_type', 'transformer')).lower()
    base_model = get_timm_model(args)

    if head_type in ['transformer', 'transformer_head']:
        return TransformerHeadClassifier(base_model, args)
    if head_type in ['linear', 'fc', 'default']:
        return _replace_classifier_head(base_model, args.n_classes)
    if head_type in ['general_gem', 'gem']:
        return GeneralWithGeM(base_model, args)
    if head_type in ['effnet_gem']:
        return EffNetwithGeM(base_model, args)
    if head_type in ['backbone', 'none']:
        return base_model

    raise ValueError(
        f"Unsupported head_type: {head_type}. "
        "Use one of ['transformer', 'linear', 'general_gem', 'effnet_gem', 'backbone']."
    )


def gradient_mask_hook(gead): 

    FROZEN_CLASS_INDICES = [1]
    
    modified_grad = gead.clone()
    modified_grad[FROZEN_CLASS_INDICES] = 0.0
    return modified_grad



def get_finetune_model(model, args): 
    
    weight_finetuning = torch.load(os.path.join(args.save_dir, args.project_name, args.finetuning_weight), map_location="cuda")
    model.load_state_dict(weight_finetuning, strict=True)

    smw_Lora = True
    if smw_Lora: 
        lora_config = LoraConfig(
            r = 4, 
            lora_alpha=16, 
            lora_dropout=0.2,
            target_modules = ['qkv', 'proj'],
            modules_to_save = ['head'] 
        )
        lora_model = get_peft_model(model, lora_config)
        
        # [수정 2] 정확히 마지막 분류기(Head)의 fc 레이어만 타겟팅합니다.
        for name, param in lora_model.named_parameters():
            
            param.requires_grad = True
            if 'head' in name and ('fc.weight' in name or 'fc.bias' in name): 
                param.register_hook(gradient_mask_hook)
                print(f"[INFO] Gradient mask hook successfully registered for: {name}")
                
    return lora_model


class GeM(nn.Module):
    def __init__(self, p=3, eps=1e-6):
        super(GeM, self).__init__()
        self.p = nn.Parameter(torch.ones(1)*p)
        self.eps = eps

    def forward(self, x):
        return self.gem(x, p=self.p, eps=self.eps)
        
    def gem(self, x, p=3, eps=1e-6):
        return F.adaptive_avg_pool2d(x.clamp(min=eps).pow(p), (1,1)).pow(1./p)
        
    def __repr__(self):
        return self.__class__.__name__ + \
                '(' + 'p=' + '{:.4f}'.format(self.p.data.tolist()[0]) + \
                ', ' + 'eps=' + str(self.eps) + ')'


class SwinWithGeM(nn.Module):
   def __init__(self, base_model, num_classes = 8):
        super().__init__()
        self.backbone = base_model
        self.gem = GeM()
        self.head = nn.Linear(base_model.num_features, base_model.num_classes)
        self.classifier = nn.Linear(base_model.num_features, num_classes)

   def forward(self, x):
        x = self.backbone.forward_features(x)  # shape: [B, C, H, W] if feature map
        x = x.permute(0,3,1,2)
        x = self.gem(x).squeeze(-1).squeeze(-1)
        return self.classifier(x)


class GeneralWithGeM(nn.Module):
   def __init__(self, model, args):
        super().__init__()
        
        # 전체 모델을 저장 (feature extraction용)
        self.backbone = model
        
        # head의 입력 차원 가져오기
        self.in_features = model.head.fc.in_features
        
        # GeM pooling 및 classifier
        self.gem = GeM()
        self.flatten = nn.Flatten()
        self.fc = nn.Linear(in_features=self.in_features, out_features=args.n_classes)

   def forward(self, x):
        x = self.backbone.forward_features(x)  # Feature extraction
        x = self.gem(x)       # GeM pooling
        x = self.flatten(x)   # Flatten
        x = self.fc(x)        # Classification
        return x


class EffNetwithGeM(nn.Module):
    def __init__(self, base_model, args):
        super(EffNetwithGeM, self).__init__()
        self.model = base_model

        in_features = self.model.classifier.in_features
        self.model.classifier = nn.Identity()
        self.model.global_pool = nn.Identity()
        self.pooling = GeM()
        self.linear = nn.Linear(in_features, args.n_classes)
        
        
    def forward(self, images):
        features = self.model(images)
        pooled_features = self.pooling(features).flatten(1)
        output = self.linear(pooled_features)
        
        return output


class TransformerHeadClassifier(nn.Module):
    def __init__(self, backbone, args, num_layers=2):
        super().__init__()
        self.backbone = backbone
        self.embed_dim = backbone.num_features # 768
        self.num_classes = args.n_classes
        self.freeze_backbone = bool(getattr(args, 'freeze_backbone', True))
        self.attn_fusion = bool(getattr(args, 'attn_fusion', False))
        
        # Freeze backbone parameters if requested.
        if self.freeze_backbone:
            for param in self.backbone.parameters():
                param.requires_grad = False
            # 백본 평가 모드로 설정 추가
            self.backbone.eval()
            print(f"[INFO] Backbone frozen. Only transformer_head and classifier will be trained.")
        
        # 새롭게 학습할 추가 Transformer 레이어 정의
        encoder_layer = nn.TransformerEncoderLayer(
            d_model=self.embed_dim,
            nhead=8, # Multi-head Attention 개수
            dim_feedforward=2048,
            dropout=0.2,
            activation='gelu',
            batch_first=True, # 입력 형태가 (Batch, Seq, Dim) 이므로 True
            norm_first=True, # LayerNorm을 나중에 적용해서 테스트 해볼 것
        )
        self.transformer_head = nn.TransformerEncoder(encoder_layer, num_layers=num_layers)

        # 토큰 단위 어텐션 풀링을 위한 경량 학습 투영층
        self.q_proj = nn.Linear(self.embed_dim, self.embed_dim)
        self.k_proj = nn.Linear(self.embed_dim, self.embed_dim)

        # cls_token과 patch_avg 비중을 조절하는 학습 게이트
        self.fusion_logit = nn.Parameter(torch.tensor(0.0))
        
        # 최종 분류기
        self.classifier = nn.Sequential(
            nn.LayerNorm(self.embed_dim),
            nn.Linear(self.embed_dim, self.num_classes)
        )

    def forward(self, x):
        # 백본에서 전체 토큰 가져오기
        if self.freeze_backbone:
            self.backbone.eval()
            with torch.no_grad():
                features = self.backbone.forward_features(x)
        else:
            features = self.backbone.forward_features(x)

        if features.ndim != 3:
            raise ValueError(
                f"Expected token features with shape [B, N, D], got {tuple(features.shape)}. "
                "This head requires a backbone that returns token sequences with a CLS token."
            )
        
        # 새 Transformer 레이어를 통과시켜 공정 도메인에 맞게 특징 재조합
        refined_features = self.transformer_head(features)
        
        # 재조합된 [CLS] 토큰(0번 인덱스) 사용
        cls_token = refined_features[:, 0, :]
        patch_tokens = refined_features[:, 1:, :]

        # # patch tokens 없는 경우 처리 (ONNX export 등에서 경고 발생 가능)
        # if not torch.jit.is_tracing() and patch_tokens.shape[1] == 0:
        #     return self.classifier(cls_token)

        if self.attn_fusion:
            # 어텐션 가중치를 이용한 패치 토큰 가중 평균
            query = self.q_proj(cls_token)  # [B, D]
            keys = self.k_proj(patch_tokens)  # [B, N-1, D]
            attn_weights = torch.einsum('bd, bnd -> bn', query, keys)  # [B, N-1]
            attn_weights = torch.softmax(attn_weights / (self.embed_dim ** 0.5), dim=1)  # [B, N-1]

            patch_avg = torch.einsum('bn, bnd -> bd', attn_weights, patch_tokens)  # [B, D]

            alpha = torch.sigmoid(self.fusion_logit)
            combined = alpha * cls_token + (1.0 - alpha) * patch_avg

        else:
            # 단순 평균 풀링
            patch_avg = patch_tokens.mean(dim=1)  # [B, D] 
            combined = cls_token + patch_avg

        return self.classifier(combined)
        
        
        # 어텐션 가중치를 이용한 패치 토큰 가중 평균
        query = self.q_proj(cls_token)  # [B, D]
        keys = self.k_proj(patch_tokens)  # [B, N-1, D]
        attn_weights = torch.einsum('bd, bnd -> bn', query, keys)  # [B, N-1]
        attn_weights = torch.softmax(attn_weights / (self.embed_dim ** 0.5), dim=1)  # [B, N-1]

        patch_avg = torch.einsum('bn, bnd -> bd', attn_weights, patch_tokens)  # [B, D]

        alpha = torch.sigmoid(self.fusion_logit)
        combined = alpha * cls_token + (1.0 - alpha) * patch_avg

        return self.classifier(combined)

    #
    def train(self, mode: bool = True):
        super().train(mode)
        if self.freeze_backbone:
            self.backbone.eval()
        return self


## FTF 탭접힘에서 contrastive learning을 활용 .. class2, 3을 잘 나누기 위함 
class SupConModel(nn.Module):

    def __init__(self, backbone, feat_dim=512, proj_dim=128, num_classes=3):
        super().__init__()
        self.backbone = backbone  # returns feature vector of size feat_dim

        # projection head for contrastive
        self.proj = nn.Sequential(
            nn.Linear(feat_dim, feat_dim),
            nn.ReLU(),
            nn.Linear(feat_dim, proj_dim)
        )
        # classification head for supervised CE (optional, can help)
        self.classifier = nn.Linear(feat_dim, num_classes)

    def forward(self, x):
        # Step 1: Backbone에서 feature 추출
        feat_map  = self.backbone.forward_features(x)  # [B, feat_dim]

        # Step 2: Global average pooling → (B, 576)
        feat = feat_map.mean(dim=[2, 3])

        # Step 3: Projection head (normalized for contrastive loss)
        proj = self.proj(feat)
        proj = F.normalize(proj, dim=1)

        # Step 4: Classification head
        logits = self.classifier(feat)

        return feat, proj, logits
    

class AIA_Adapter(nn.Module):
    def __init__(self, dim):
        super().__init__()
        self.fc = nn.Sequential(
            nn.Linear(dim, dim),
            nn.ReLU(inplace=True),
            nn.Linear(dim, dim)
        )

    def forward(self, x):
        return x + self.fc(x)


class AIA_MidfusionModel(nn.Module):
    def __init__(self, backbone, num_classes=5):
        super().__init__()
        # EfficientNetV2-m backbone (shared)
        self.backbone = backbone
        
        self.backbone.global_pool = nn.Sequential(
            nn.AdaptiveAvgPool2d(1),
            nn.Flatten(1))
        self.feature_dim = self.backbone.num_features

        # 5개의 Adapter 선언
        self.adapter1 = AIA_Adapter(self.feature_dim)
        self.adapter2 = AIA_Adapter(self.feature_dim)
        self.adapter3 = AIA_Adapter(self.feature_dim)
        self.adapter4 = AIA_Adapter(self.feature_dim)
        self.adapter5 = AIA_Adapter(self.feature_dim)

        # Concat fusion → classifier
        self.classifier = nn.Linear(self.feature_dim * 5, num_classes)

    def forward(self, x):
        """
        x: [B, 5, C, H, W]
        """
        f1 = self.adapter1(self.backbone(x[:, 0]))
        f2 = self.adapter2(self.backbone(x[:, 1]))
        f3 = self.adapter3(self.backbone(x[:, 2]))
        f4 = self.adapter4(self.backbone(x[:, 3]))
        f5 = self.adapter5(self.backbone(x[:, 4]))

        fused = torch.cat([f1, f2, f3, f4, f5], dim=1)  # [B, 5*feature_dim]
        out = self.classifier(fused)
        return out
