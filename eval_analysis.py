import os
import math
import json
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


# run_schedule.py 스타일: 기본 config + 필요한 값만 override
ANALYSIS_JOB = {
    'config': './config/SURFACE_ANODE_classification.yaml',
    'overrides': {
        'project_name': '0804_SURFACE_ANODE_DiNO_CLS_FA05',
        'token_fusion': 'cls_only',
        'stage1_loss_mode': 'focal',
        'focal_alpha': [0.5, 0.5, 0.5, 0.5, 0.5, 0.5, 0.5, 0.5, 0.5, 0.5, 0.5],
    },
    # 특정 체크포인트 파일명 지정 시 해당 .pth만 분석 (예: 'f1_score0.95_Loss0.12_epoch30.pth')
    # 비워두면 project_name 디렉토리의 모든 .pth를 분석
    # 'target_pth_name': '',
    'target_pth_name': 'f1_score0.8667_Loss0.3926265_epoch39.pth',
    # train_arguments.json 동기화 사용 여부
    'use_train_arguments_sync': True,
    # 동기화 시에도 로컬 실행 경로는 유지(덮어쓰기 방지)
    'preserve_path_keys': ['save_dir', 'model_dir', 'train_dir', 'test_dir', 'device'],
}


def apply_overrides(args, overrides):
    for key, value in (overrides or {}).items():
        setattr(args, key, value)


def sync_args_from_training_artifact(args, logger, preserve_keys=None):
    """학습 시 저장된 train_arguments.json을 불러와 추론 설정 불일치를 방지한다."""
    preserve_keys = preserve_keys or []
    preserved = {k: getattr(args, k) for k in preserve_keys if hasattr(args, k)}

    train_args_path = os.path.join(args.save_dir, args.project_name, 'train_arguments.json')
    if not os.path.exists(train_args_path):
        logger.warning(f'train_arguments.json not found: {train_args_path}')
        return

    with open(train_args_path, 'r', encoding='utf-8') as f:
        trained_args = json.load(f)

    if not isinstance(trained_args, dict):
        logger.warning(f'Invalid train_arguments.json format: {train_args_path}')
        return

    for key, value in trained_args.items():
        setattr(args, key, value)

    # 로컬 실행 경로/디바이스는 유지
    for key, value in preserved.items():
        setattr(args, key, value)

    logger.info(f'Synced args from training artifact: {train_args_path}')


def find_latest_file(directory, pattern):
    """디렉토리 내 패턴 파일 중 최신 수정 파일 경로 반환"""
    import glob

    candidates = glob.glob(os.path.join(directory, pattern))
    if not candidates:
        return None
    return max(candidates, key=os.path.getmtime)


def resolve_model_jobs(project_root, target_pth_name, logger):
    """분석 대상 모델/CSV 목록 결정.

    - target_pth_name 지정: 해당 .pth 1개만
    - target_pth_name 미지정: 디렉토리 내 모든 .pth
    """
    import glob

    target_name = str(target_pth_name or '').strip()

    if target_name:
        if not target_name.lower().endswith('.pth'):
            target_name = f'{target_name}.pth'

        model_path = os.path.join(project_root, target_name)
        if not os.path.exists(model_path):
            raise RuntimeError(f'Target model not found: {model_path}')

        csv_path = model_path + '.csv'
        csv_files = [csv_path] if os.path.exists(csv_path) else []
        if not csv_files:
            logger.warning(f'CSV not found for target model: {csv_path}')

        return [(model_path, csv_files)]

    model_paths = sorted(glob.glob(os.path.join(project_root, '*.pth')))
    if not model_paths:
        raise RuntimeError(f'No .pth file found in {project_root}')

    jobs = []
    for model_path in model_paths:
        csv_path = model_path + '.csv'
        if os.path.exists(csv_path):
            jobs.append((model_path, [csv_path]))
        else:
            logger.warning(f'Skipping model without matched CSV: {model_path}')

    if not jobs:
        raise RuntimeError(f'No matched .csv found for .pth files in {project_root}')

    return jobs


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

    for idx, item in enumerate(report_items_mismatch, start=1):
        lines.extend([
            '    <div class="card">',
            f'      <div class="title"><strong>오분류 {idx}. csv_file_path:</strong> {item["file_path"]}</div>',
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

    for idx, item in enumerate(report_items_correct, start=1):
        lines.extend([
            '    <div class="card">',
            f'      <div class="title"><strong>정분류 {idx}. csv_file_path:</strong> {item["file_path"]}</div>',
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
    """token_fusion 방식에 맞는 패치 중요도 히트맵 계산.
    - attn / wt_concat : 학습된 q/k projection 어텐션 가중치 직접 사용 (forward와 동일)
    - concat/sum : GradCAM — backbone 출력 patch features를 leaf 텐서로 만들어
                   예측 클래스 score에 대한 gradient를 직접 수집
    """
    if not hasattr(model, 'backbone') or not hasattr(model, 'transformer_head'):
        raise RuntimeError('Model does not expose transformer attention internals.')

    token_fusion = str(getattr(model, 'token_fusion', 'sum')).lower()
    model.eval()

    # ── backbone forward (항상 no_grad, frozen) ──────────────────────────
    with torch.no_grad():
        features = model.backbone.forward_features(image_tensor.unsqueeze(0))

    if features.ndim != 3:
        raise RuntimeError(f'Expected token features [B, N, D], got {tuple(features.shape)}')

    if token_fusion in ['attn', 'wt_concat'] and hasattr(model, 'q_proj') and hasattr(model, 'k_proj'):
        # ── attn / wt_concat 모드: forward와 동일한 연산, gradient 불필요 ──
        with torch.no_grad():
            refined = model.transformer_head(features)
            cls_token  = refined[:, 0, :]
            patch_tokens = refined[:, 1:, :]

            if patch_tokens.shape[1] == 0:
                raise RuntimeError('No patch tokens found.')

            q    = model.q_proj(cls_token)
            k    = model.k_proj(patch_tokens)
            attn = torch.einsum('bd,bnd->bn', q, k) / math.sqrt(model.embed_dim)
            attn = torch.softmax(attn, dim=1)

    else:
        # ── concat / sum 모드: GradCAM ─────────────────────────────────────
        # features를 leaf 텐서로 만들면 backward() 후 .grad에 gradient가 채워짐
        # (hook 불필요, slice view 문제 없음)
        features_leaf = features.detach().requires_grad_(True)  # leaf tensor

        with torch.enable_grad():
            refined      = model.transformer_head(features_leaf)
            cls_token    = refined[:, 0, :]
            patch_tokens = refined[:, 1:, :]

            if patch_tokens.shape[1] == 0:
                raise RuntimeError('No patch tokens found.')

            if token_fusion == 'concat':
                patch_avg = patch_tokens.mean(dim=1)
                combined  = torch.cat([cls_token, patch_avg], dim=1)
                combined  = model.concat_fusion(combined)
            else:  # sum
                patch_avg = patch_tokens.mean(dim=1)
                combined  = cls_token + patch_avg

            logits     = model.classifier(combined)
            pred_class = logits.argmax(dim=1).item()
            logits[0, pred_class].backward()

        # leaf tensor이므로 .grad가 항상 채워짐
        # features_leaf: [1, N+1, D]  (index 0 = CLS, 1: = patches)
        grads       = features_leaf.grad   # [1, N+1, D]
        patch_grads = grads[:, 1:, :]      # CLS 제외: [1, N, D]

        weights = patch_grads.mean(dim=-1)  # [1, N]
        attn    = torch.relu(weights)
        if attn.sum().item() == 0:          # relu로 전부 소멸 시 절댓값 fallback
            attn = patch_grads.abs().mean(dim=-1)
        attn = attn / (attn.sum() + 1e-8)

    # ── 공통: 2D 그리드로 변환 후 이미지 크기로 업샘플 ──────────────────
    num_patches = attn.shape[1]
    patch_side  = int(math.sqrt(num_patches))
    if patch_side * patch_side != num_patches:
        raise RuntimeError(f'Unexpected patch count {num_patches}, not square.')

    attn_map = attn.view(1, 1, patch_side, patch_side)
    attn_map = F.interpolate(attn_map, size=output_size, mode='bilinear', align_corners=False)
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
    config_path = ANALYSIS_JOB.get('config', './config/SURFACE_ANODE_classification.yaml')
    overrides = ANALYSIS_JOB.get('overrides', {})
    target_pth_name = ANALYSIS_JOB.get('target_pth_name', '')
    use_train_sync = bool(ANALYSIS_JOB.get('use_train_arguments_sync', True))
    preserve_path_keys = ANALYSIS_JOB.get('preserve_path_keys', ['save_dir', 'model_dir', 'train_dir', 'test_dir', 'device'])

    args = argument()
    args.load(config_path)
    apply_overrides(args, overrides)

    # project_name 기준으로 학습 당시 인자 동기화 (구조 mismatch 방지)
    if use_train_sync:
        tmp_logger = setup_logger('eval-analysis-bootstrap')
        sync_args_from_training_artifact(args, tmp_logger, preserve_keys=preserve_path_keys)
    # 의도적으로 지정한 값은 train_arguments 로드 후에도 우선 적용
    apply_overrides(args, overrides)

    logger = setup_logger('eval-analysis', log_file=f'{args.save_dir}/{args.project_name}/analysis.log')
    logger.info(f'Loaded config: {config_path}')
    logger.info(f'Applied overrides: {overrides}')
    logger.info(f'use_train_arguments_sync: {use_train_sync}')
    logger.info(f'preserve_path_keys: {preserve_path_keys}')
    logger.info(f'Pretrained model: {args.pretrained}')

    filenames_total, label_total = collect_test_data(args)
    if args.datasets_name.lower() == 'smw':
        raise RuntimeError('SMW는 이 스크립트에서 지원하지 않습니다.')

    dataset = datasets.SURFACEDataset(filenames_total, label_total, args, mode='valid')
    path_to_index = {path: idx for idx, path in enumerate(dataset.filename_list)}

    project_root = os.path.join(args.save_dir, args.project_name)

    model_jobs = resolve_model_jobs(project_root, target_pth_name, logger)
    if str(target_pth_name).strip():
        logger.info(f'Target mode: single checkpoint | {target_pth_name}')
    else:
        logger.info(f'Target mode: all checkpoints | count={len(model_jobs)}')

    save_root_base = os.path.join(args.save_dir, args.project_name, 'analysis_attention')
    os.makedirs(save_root_base, exist_ok=True)

    device = torch.device(args.device)

    for model_path, csv_files in model_jobs:
        model_name = os.path.splitext(os.path.basename(model_path))[0]
        logger.info(f'\n========== Processing model: {model_path} ==========')

        model = load_model(args, model_path, device, strict=False)
        model.to(device)
        model.eval()

        if str(getattr(model, 'token_fusion', 'sum')).lower() == 'attn' and hasattr(model, 'fusion_logit'):
            alpha = torch.sigmoid(model.fusion_logit.detach()).item()
            logger.info(f'Fusion weight (alpha): {alpha:.4f}')
        else:
            logger.info('Fusion weight (alpha): not applicable for this token_fusion mode')

        if not csv_files:
            logger.warning(f'No CSV to analyze for model: {model_path}')
            continue

        model_save_root = os.path.join(save_root_base, model_name)
        os.makedirs(model_save_root, exist_ok=True)

        for csv_path in csv_files:
            csv_name = os.path.splitext(os.path.basename(csv_path))[0]
            save_root = model_save_root

            logger.info(f'\n========== Processing CSV: {csv_path} ==========')
            logger.info(f'Save folder: {save_root}')

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

                original_name = f'mismatch_{base_name}_orig.png'
                original_path = os.path.join(save_root, original_name)
                original_image.save(original_path)

                out_name = f'mismatch_{base_name}_attn.png'
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

                original_name = f'correct_{base_name}_orig.png'
                original_path = os.path.join(save_root, original_name)
                original_image.save(original_path)

                out_name = f'correct_{base_name}_attn.png'
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
            report_path = os.path.join(save_root, f'analysis_report_{csv_name}.html')
            build_html_report(report_path, report_items_mismatch, report_items_correct)
            logger.info(f'Saved analysis report: {report_path}')

    logger.info('\n========== Attention visualization finished. ==========')

if __name__ == '__main__':
    main()
