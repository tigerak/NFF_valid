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


def find_latest_file(directory, pattern):
    """디렉토리 내 패턴 파일 중 최신 수정 파일 경로 반환"""
    import glob

    candidates = glob.glob(os.path.join(directory, pattern))
    if not candidates:
        return None
    return max(candidates, key=os.path.getmtime)


def build_html_report(report_path, report_items_mismatch, report_items_correct):
    """원본/히트맵/메타 정보를 포함한 HTML 리포트 저장 (mismatch와 correct 분리)"""
    def _fmt(v):
        if isinstance(v, float):
            return f"{v:.6f}"
        return str(v)

    lines = [
        '<!DOCTYPE html>',
        '<html lang="ko">',
        '<head>',
        '  <meta charset="UTF-8">',
        '  <meta name="viewport" content="width=device-width, initial-scale=1.0">',
        '  <title>Evaluation Analysis Report</title>',
        '  <style>',
        '    body { font-family: Arial, sans-serif; margin: 20px; background: #f7f8fa; color: #1f2937; }',
        '    .section { margin-bottom: 40px; }',
        '    .section-title { font-size: 20px; font-weight: bold; margin: 30px 0 20px 0; padding: 10px; background: #e0e7ff; border-left: 4px solid #4f46e5; }',
        '    .card { background: #ffffff; border: 1px solid #e5e7eb; border-radius: 10px; padding: 16px; margin-bottom: 20px; }',
        '    .title { font-size: 14px; margin-bottom: 10px; word-break: break-all; }',
        '    .images { display: flex; gap: 16px; flex-wrap: wrap; }',
        '    .img-wrap { min-width: 280px; }',
        '    .img-wrap h4 { margin: 0 0 8px 0; font-size: 13px; color: #374151; }',
        '    img { max-width: 420px; width: 100%; border: 1px solid #d1d5db; border-radius: 8px; }',
        '    table { border-collapse: collapse; width: 100%; margin-top: 12px; }',
        '    th, td { border: 1px solid #e5e7eb; padding: 6px 8px; font-size: 12px; text-align: left; }',
        '    th { background: #f3f4f6; width: 240px; }',
        '  </style>',
        '</head>',
        '<body>',
        '  <h2>Evaluation Analysis Report</h2>',
        f'  <p>Mismatch (오분류): {len(report_items_mismatch)} | Correct (정분류): {len(report_items_correct)} | Total: {len(report_items_mismatch) + len(report_items_correct)}</p>',
    ]

    # Mismatch 섹션
    lines.extend([
        '  <div class="section">',
        '    <div class="section-title">🔴 Mismatch Samples (True Label ≠ Predicted Label)</div>',
    ])

    for item in report_items_mismatch:
        lines.extend([
            '    <div class="card">',
            f'      <div class="title"><strong>file_path:</strong> {item["file_path"]}</div>',
            '      <div class="images">',
            '        <div class="img-wrap">',
            '          <h4>Original</h4>',
            f'          <img src="{item["original_rel"]}" alt="original">',
            '        </div>',
            '        <div class="img-wrap">',
            '          <h4>Heatmap</h4>',
            f'          <img src="{item["heatmap_rel"]}" alt="heatmap">',
            '        </div>',
            '      </div>',
            '      <table>',
            '        <tr><th>key</th><th>value</th></tr>',
        ])

        for key, value in item['meta'].items():
            lines.append(f'        <tr><td>{key}</td><td>{_fmt(value)}</td></tr>')

        lines.extend([
            '      </table>',
            '    </div>',
        ])

    lines.append('  </div>')

    # Correct 섹션
    lines.extend([
        '  <div class="section">',
        '    <div class="section-title">🟢 Correct Samples (True Label = Predicted Label)</div>',
    ])

    for item in report_items_correct:
        lines.extend([
            '    <div class="card">',
            f'      <div class="title"><strong>file_path:</strong> {item["file_path"]}</div>',
            '      <div class="images">',
            '        <div class="img-wrap">',
            '          <h4>Original</h4>',
            f'          <img src="{item["original_rel"]}" alt="original">',
            '        </div>',
            '        <div class="img-wrap">',
            '          <h4>Heatmap</h4>',
            f'          <img src="{item["heatmap_rel"]}" alt="heatmap">',
            '        </div>',
            '      </div>',
            '      <table>',
            '        <tr><th>key</th><th>value</th></tr>',
        ])

        for key, value in item['meta'].items():
            lines.append(f'        <tr><td>{key}</td><td>{_fmt(value)}</td></tr>')

        lines.extend([
            '      </table>',
            '    </div>',
        ])

    lines.append('  </div>')

    lines.extend(['</body>', '</html>'])

    with open(report_path, 'w', encoding='utf-8') as f:
        f.write('\n'.join(lines))


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
    """token_fusion 방식에 맞는 패치 중요도 히트맵 계산."""
    if not hasattr(model, 'backbone') or not hasattr(model, 'transformer_head'):
        raise RuntimeError('Model does not expose transformer attention internals.')

    token_fusion = str(getattr(model, 'token_fusion', 'sum')).lower()

    model.eval()
    with torch.no_grad():
        features = model.backbone.forward_features(image_tensor.unsqueeze(0))

    if features.ndim != 3:
        raise RuntimeError(f'Expected token features [B, N, D], got {tuple(features.shape)}')

    refined = model.transformer_head(features)
    cls_token = refined[:, 0, :]
    patch_tokens = refined[:, 1:, :]

    if patch_tokens.shape[1] == 0:
        raise RuntimeError('No patch tokens found.')

    if token_fusion == 'attn' and hasattr(model, 'q_proj') and hasattr(model, 'k_proj'):
        q = model.q_proj(cls_token)
        k = model.k_proj(patch_tokens)
        attn = torch.einsum('bd,bnd->bn', q, k) / math.sqrt(model.embed_dim)
        attn = torch.softmax(attn, dim=1)
    elif token_fusion == 'concat':
        cls_norm = F.normalize(cls_token, dim=1)
        patch_norm = F.normalize(patch_tokens, dim=2)
        attn = torch.einsum('bd,bnd->bn', cls_norm, patch_norm)
        attn = torch.softmax(attn, dim=1)
    else:
        patch_score = patch_tokens.norm(dim=-1)
        attn = torch.softmax(patch_score, dim=1)

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
    return attn_map[0, 0].detach().cpu().numpy()


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
    """Transformer attention 시각화 실행 스크립트 - 모든 CSV 파일 처리"""
    import glob
    
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

    project_root = os.path.join(args.save_dir, args.project_name)

    # 분석 대상 모델(.pth) 자동 선택: 최신 파일
    model_path = find_latest_file(project_root, '*.pth')
    if model_path is None:
        raise RuntimeError(f'No .pth file found in {project_root}')
    logger.info(f'Selected model: {model_path}')

    # 모든 CSV 파일 찾기
    csv_files = sorted(glob.glob(os.path.join(project_root, '*.csv')))
    if not csv_files:
        raise RuntimeError(f'No evaluation CSV found in {project_root}')
    
    logger.info(f'Found {len(csv_files)} CSV files to process')

    device = torch.device(args.device)
    model = load_model(args, model_path, device, strict=False)
    model.to(device)
    model.eval()

    if str(getattr(model, 'token_fusion', 'sum')).lower() == 'attn' and hasattr(model, 'fusion_logit'):
        alpha = torch.sigmoid(model.fusion_logit.detach()).item()
        logger.info(f'Fusion weight (alpha): {alpha:.4f}')
    else:
        logger.info('Fusion weight (alpha): not applicable for this token_fusion mode')

    save_root = os.path.join(args.save_dir, args.project_name, 'analysis_attention')
    os.makedirs(save_root, exist_ok=True)

    # 각 CSV 파일에 대해 처리
    for csv_path in csv_files:
        logger.info(f'\n========== Processing CSV: {csv_path} ==========')
        
        eval_df = pd.read_csv(csv_path)

        required_base_cols = {'file_path', 'true_label_name', 'pred_label_name'}
        missing_cols = [c for c in required_base_cols if c not in eval_df.columns]
        if missing_cols:
            logger.error(f'Missing required columns in CSV {csv_path}: {missing_cols}')
            continue

        confidence_cols = [c for c in eval_df.columns if c.startswith('confidence_')]
        if len(confidence_cols) == 0:
            logger.warning(f'No confidence_* columns found in CSV {csv_path}. Skipping.')
            continue

        # Mismatch (true_label_name != pred_label_name) 샘플
        mismatch_df = eval_df[eval_df['true_label_name'] != eval_df['pred_label_name']].copy()
        logger.info(f'Mismatch samples found: {len(mismatch_df)}')

        # Correct (true_label_name == pred_label_name) 샘플
        correct_df = eval_df[eval_df['true_label_name'] == eval_df['pred_label_name']].copy()
        logger.info(f'Correct samples found: {len(correct_df)}')

        # 메타 컬럼 선택: true_label_name, pred_label_name, confidence_*
        meta_cols = ['true_label_name', 'pred_label_name'] + confidence_cols

        report_items_mismatch = []
        report_items_correct = []

        # Mismatch 처리
        for _, row in mismatch_df.iterrows():
            file_path = row['file_path']
            if file_path not in path_to_index:
                logger.warning(f'CSV path not found in dataset list: {file_path}')
                continue

            image_tensor, label, path = dataset[path_to_index[file_path]]
            image_tensor = image_tensor.to(device)

            with torch.no_grad():
                outputs = model(image_tensor.unsqueeze(0))
                probs = torch.softmax(outputs, dim=1)
                pred = torch.argmax(probs, dim=1).item()

            attention_map = compute_transformer_attention(model, image_tensor, tuple(args.img_size))
            original_image = Image.open(path).convert('RGB').resize(tuple(args.img_size))
            overlay = overlay_attention(original_image, attention_map, alpha=0.5)

            base_name = os.path.splitext(os.path.basename(path))[0]
            csv_name = os.path.splitext(os.path.basename(csv_path))[0]

            original_name = f'{csv_name}_mismatch_{base_name}_orig.png'
            original_path = os.path.join(save_root, original_name)
            original_image.save(original_path)

            out_name = f'{csv_name}_mismatch_{base_name}_attn.png'
            out_path = os.path.join(save_root, out_name)
            overlay.save(out_path)
            logger.info(f'Saved mismatch attention overlay: {out_path}')

            report_items_mismatch.append({
                'file_path': path,
                'original_rel': original_name,
                'heatmap_rel': out_name,
                'meta': {k: row[k] for k in meta_cols if k in row},
            })

        # Correct 처리 - 클래스 별로 5개씩 랜덤 샘플링
        correct_samples = []
        for label_name in correct_df['pred_label_name'].unique():
            label_df = correct_df[correct_df['pred_label_name'] == label_name]
            sample_size = min(5, len(label_df))
            sampled = label_df.sample(n=sample_size, random_state=42)
            correct_samples.append(sampled)
        
        correct_df_sampled = pd.concat(correct_samples, ignore_index=True) if correct_samples else pd.DataFrame()
        logger.info(f'Sampled {len(correct_df_sampled)} correct samples (5 per class max)')

        for _, row in correct_df_sampled.iterrows():
            file_path = row['file_path']
            if file_path not in path_to_index:
                logger.warning(f'CSV path not found in dataset list: {file_path}')
                continue

            image_tensor, label, path = dataset[path_to_index[file_path]]
            image_tensor = image_tensor.to(device)

            with torch.no_grad():
                outputs = model(image_tensor.unsqueeze(0))
                probs = torch.softmax(outputs, dim=1)
                pred = torch.argmax(probs, dim=1).item()

            attention_map = compute_transformer_attention(model, image_tensor, tuple(args.img_size))
            original_image = Image.open(path).convert('RGB').resize(tuple(args.img_size))
            overlay = overlay_attention(original_image, attention_map, alpha=0.5)

            base_name = os.path.splitext(os.path.basename(path))[0]
            csv_name = os.path.splitext(os.path.basename(csv_path))[0]

            original_name = f'{csv_name}_correct_{base_name}_orig.png'
            original_path = os.path.join(save_root, original_name)
            original_image.save(original_path)

            out_name = f'{csv_name}_correct_{base_name}_attn.png'
            out_path = os.path.join(save_root, out_name)
            overlay.save(out_path)
            logger.info(f'Saved correct attention overlay: {out_path}')

            report_items_correct.append({
                'file_path': path,
                'original_rel': original_name,
                'heatmap_rel': out_name,
                'meta': {k: row[k] for k in meta_cols if k in row},
            })

        # CSV별 HTML 리포트 저장
        csv_name = os.path.splitext(os.path.basename(csv_path))[0]
        report_path = os.path.join(save_root, f'analysis_report_{csv_name}.html')
        build_html_report(report_path, report_items_mismatch, report_items_correct)
        logger.info(f'Saved analysis report: {report_path}')

    logger.info('\n========== Attention visualization finished. ==========')

if __name__ == '__main__':
    main()
