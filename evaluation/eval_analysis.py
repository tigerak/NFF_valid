import os
import math
import numpy as np
import pandas as pd
import torch
import torch.nn.functional as F
import cv2
from PIL import Image

os.environ['KMP_DUPLICATE_LIB_OK'] = 'True'

from argments import argument
from utils import datasets
from dataset import data_setup

from evaluation.eval import setup_logger
from evaluation.base_engine import load_model


def collect_test_data(args):
    """테스트 데이터 수집
    Args:
        args: 설정 객체
    Returns:
        filenames_total: 이미지 경로 리스트
        label_total: 레이블 리스트
    """
    image_data = {
        label: data_setup.load_images_by_label(args.test_dir, label, mode='test')
        for label in args.labels
    }

    filenames_total = []
    label_total = []

    for label, images in image_data.items():
        filenames_total.extend(images)
        label_total.extend([label] * len(images))

    print(f"Total: {len(filenames_total)}")
    df_labels = pd.DataFrame({'label': label_total})

    print("\n=== Test Label Distribution ===")
    print(df_labels['label'].value_counts())

    return filenames_total, label_total


def compute_transformer_attention(model, image_tensor, output_size):
    """Transformer 기반 CLS-패치 어텐션 히트맵 계산

    이 함수는 TransformerHeadClassifier에서 CLS 토큰과 패치 토큰
    간의 유사도를 계산하여 입력 이미지에 대한 어텐션 맵을 얻습니다.

    Args:
        model: TransformerHeadClassifier 형태의 모델
        image_tensor: 단일 이미지 텐서, shape [C, H, W]
        output_size: 히트맵을 업샘플할 최종 출력 크기 (H, W)

    Returns:
        numpy.ndarray: [H, W] 크기의 정규화된 어텐션 맵
    """
    # 이 스크립트는 모델이 backbone과 transformer_head를 노출할 때 동작합니다.
    # TransformerHeadClassifier는 backbone.forward_features()로 token sequence를
    # 얻고, transformer_head를 거쳐 최종 분류 결과를 만듭니다.
    if not hasattr(model, 'backbone') or not hasattr(model, 'transformer_head'):
        raise RuntimeError('Model does not expose transformer attention internals.')

    model.eval()
    # 어텐션 맵을 만드는 과정에서는 gradient가 필요 없으므로 no_grad를 사용
    with torch.no_grad():
        # image_tensor는 [C, H, W] 형태이므로 배치 차원을 추가해서 [1, C, H, W]로 만듬
        features = model.backbone.forward_features(image_tensor.unsqueeze(0))

    # DINOv2/ViT backbone은 토큰 시퀀스를 반환해야 합니다.
    # expected shape: [B, N, D], B는 배치 크기, N은 토큰 개수, D는 임베딩 차원
    if features.ndim != 3:
        raise RuntimeError(f'Expected token features [B, N, D], got {tuple(features.shape)}')

    # transformer_head를 통과시키면 CLS 토큰과 patch token이 모두 포함된 토큰 시퀀스가 나옵니다.
    refined = model.transformer_head(features)
    # CLS 토큰은 첫 번째 토큰이며 전체 이미지 특징을 요약합니다.
    cls_token = refined[:, 0, :]
    # 나머지 토큰이 실제 이미지 패치를 나타냅니다.
    patch_tokens = refined[:, 1:, :]

    # CLS 토큰과 patch 토큰을 각각 query, key로 투사합니다.
    # 이 때 query는 CLS 토큰, key는 patch 토큰이며, 두 벡터의 내적을 통해
    # CLS가 각 패치에 얼마나 주의를 기울였는지 계산합니다.
    q = model.q_proj(cls_token)
    k = model.k_proj(patch_tokens)

    # q와 k의 내적 결과는 [B, N-1] 형태의 attention score입니다.
    # 크기 보정을 위해 sqrt(embed_dim)으로 나누고 softmax를 취해 확률로 변환합니다.
    attn = torch.einsum('bd,bnd->bn', q, k) / math.sqrt(model.embed_dim)
    attn = torch.softmax(attn, dim=1)

    # attention score는 패치 개수만큼의 길이를 가집니다.
    num_patches = attn.shape[1]
    patch_side = int(math.sqrt(num_patches))
    if patch_side * patch_side != num_patches:
        raise RuntimeError(f'Unexpected patch count {num_patches}, not square.')

    # 1차원 패치 어텐션을 2D 패치 그리드로 변환합니다.
    # 예: 196개 패치 -> 14x14 그리드
    attn_map = attn.view(1, 1, patch_side, patch_side)

    # 최종 이미지 크기로 업샘플하여 히트맵을 만듭니다.
    # bilinear interpolation을 사용해 부드러운 맵 생성
    attn_map = F.interpolate(attn_map, size=output_size, mode='bilinear', align_corners=False)

    # 결과를 CPU numpy로 변환해서 반환
    return attn_map[0, 0].cpu().numpy()


def overlay_attention(image: Image.Image, attention_map: np.ndarray, alpha=0.5):
    image_np = np.array(image.convert('RGB'))
    attention_norm = attention_map - attention_map.min()
    if attention_norm.max() > 0:
        attention_norm = attention_norm / attention_norm.max()
    heatmap = np.uint8(attention_norm * 255)
    heatmap_color = cv2.applyColorMap(heatmap, cv2.COLORMAP_JET)
    heatmap_color = cv2.cvtColor(heatmap_color, cv2.COLOR_BGR2RGB)
    overlay = cv2.addWeighted(image_np, 1.0 - alpha, heatmap_color, alpha, 0)
    return Image.fromarray(overlay)


def main():
    """Transformer attention 시각화 실행 스크립트"""
    config_path = './config/SURFACE_ANODE_classification.yaml'
    # config_path = './config/SURFACE_CATHODE_classification.yaml'
    # config_path = './config/SMW_classification.yaml'

    args = argument()
    args.load(config_path)

    logger = setup_logger('eval-analysis', log_file=f'{args.save_dir}/{args.project_name}/analysis.log')
    logger.info(f'Loaded config: {config_path}')
    logger.info(f'Pretrained model: {args.pretrained}')

    filenames_total, label_total = collect_test_data(args)
    if args.datasets_name.lower() == 'smw':
        raise RuntimeError('SMW는 이 스크립트에서 지원하지 않습니다.')

    dataset = datasets.SURFACEDataset(filenames_total, label_total, args, mode='valid')
    path_to_index = {path: idx for idx, path in enumerate(dataset.filename_list)}

    selected_paths = [
        # r'D:\path\to\your_image1.bmp',
        # r'D:\path\to\your_image2.bmp',
    ]

    ### CSV 확인하고 이미지 경로를 selected_paths에 추가하는 코드 추가 작성 ###

    if len(selected_paths) == 0:
        raise RuntimeError('selected_paths에 분석할 이미지 경로를 추가하세요.')

    # model_path = os.path.join(args.save_dir, args.project_name, args.finetuning_weight)
    model_path = r""
    device = torch.device(args.device)
    model = load_model(args, model_path, device)
    model.to(device)
    model.eval()

    save_root = os.path.join(args.save_dir, args.project_name, 'analysis_attention')
    os.makedirs(save_root, exist_ok=True)

    for selected_path in selected_paths:
        if selected_path not in path_to_index:
            logger.warning(f'Path not found in dataset list: {selected_path}')
            continue

        image_tensor, label, path = dataset[path_to_index[selected_path]]
        image_tensor = image_tensor.to(device)

        with torch.no_grad():
            outputs = model(image_tensor.unsqueeze(0))
            probs = torch.softmax(outputs, dim=1)
            pred = torch.argmax(probs, dim=1).item()
            logger.info(f'Image={path} true={args.labels[label]} pred={args.labels[pred]}')

        attention_map = compute_transformer_attention(model, image_tensor, tuple(args.img_size))
        original_image = Image.open(path).convert('RGB').resize(tuple(args.img_size))
        overlay = overlay_attention(original_image, attention_map, alpha=0.5)

        out_name = os.path.splitext(os.path.basename(path))[0] + '_attn.png'
        out_path = os.path.join(save_root, out_name)
        overlay.save(out_path)
        logger.info(f'Saved attention overlay: {out_path}')

    logger.info('Attention visualization finished.')

if __name__ == '__main__':
    main()
