"""
스케줄 트레이닝 러너
====================
YAML 파일을 기반으로 옵션만 오버라이드해서 여러 학습을 순서대로 실행합니다.

사용법:
    python run_schedule.py

SCHEDULE 리스트에 실행할 작업을 정의하세요.
각 항목은 아래 키를 가집니다:
    config   : 기준 YAML 파일 경로 (필수)
    overrides: dict, YAML 값을 덮어쓸 옵션 (선택)

특별한 경우 평가만 실행하려면 파일 하단의 __main__ 블록에서
main() 대신 main_eval_only()를 호출하세요.
"""

import os
import glob
import gc
import time
import traceback
import logging
from datetime import datetime

import torch
from torch.utils.data import DataLoader

from argments import argument
from utils import get_models, datasets, schedulers, training
from dataset import data_setup
from evaluation.base_engine import setup_logger
from evaluation.eval import EvalManager

# ===========================================================================
# 여기에 실행할 작업 목록을 정의하세요
# ===========================================================================
TODAY = datetime.now().strftime('%m%d')

# Stage1(ArcFace) -> Stage2 파이프라인에서 동일한 실험명을 공유하도록 상수화
STAGE1_SCARC_PROJECT = f'SURFACE_ANODE_DiNO_Concat_SCArc_{TODAY}'

SCHEDULE = [
    # ----- ANODE: sum -----
    {
        'config': './config/SURFACE_ANODE_classification.yaml',
        'overrides': {
            'project_name': f'SURFACE_ANODE_DiNO_Sum_{TODAY}',
            'token_fusion': 'sum',
            'stage1_loss_mode': 'focal',
        },
    },

    # # ----- ANODE: weight concat -----
    # {
    #     'config': './config/SURFACE_ANODE_classification.yaml',
    #     'overrides': {
    #         'project_name': f'SURFACE_ANODE_DiNO_Concat_{TODAY}',
    #         'token_fusion': 'wt_concat',
    #         'focal_alpha': [1., 1., 1., 1., 1., 1., 1., 1., 1., 1., 1.]
    #     },
    # },

    # # ----- ANODE: Attention / focal alpha 1. -----
    # {
    #     'config': './config/SURFACE_ANODE_classification.yaml',
    #     'overrides': {
    #         'project_name': f'SURFACE_ANODE_DiNO_Attn_{TODAY}',
    #         'token_fusion': 'attn',
    #         'focal_alpha': [1., 1., 1., 1., 1., 1., 1., 1., 1., 1., 1.]
    #     },
    # },

    # # ----- ANODE: Attention / focal alpha .5 -----
    # {
    #     'config': './config/SURFACE_ANODE_classification.yaml',
    #     'overrides': {
    #         'project_name': f'SURFACE_ANODE_DiNO_Attn_FA05_{TODAY}',
    #         'token_fusion': 'attn',
    #         'focal_alpha': [.5, .5, .5, .5, .5, .5, .5, .5, .5, .5, .5]
    #     },
    # },

    
    # # ----- ANODE: Attention / focal alpha 1. / Mixup -----
    # {
    #     'config': './config/SURFACE_ANODE_classification.yaml',
    #     'overrides': {
    #         'project_name': f'SURFACE_ANODE_DiNO_Attn_FA1_Mixup_{TODAY}',
    #         'token_fusion': 'attn',
    #         'focal_alpha': [1., 1., 1., 1., 1., 1., 1., 1., 1., 1., 1.],
    #         'use_mixup': True,
    #         'mixup_alpha': 1.0
    #     },
    # },
    
    # ----- ANODE: weight concat / focal alpha .5 -----
    # ======================================================================
    # 비교용 예시 2: Stage-1 / Sub-center ArcFace
    # - cls + patch token concat 후 Sub-center ArcFace
    # - mixup / cutmix 자동 비활성화
    # - 4 sub-centers, m=0.3, s=64
    # ======================================================================
    {
        'config': './config/SURFACE_ANODE_classification.yaml',
        'overrides': {
            'project_name': STAGE1_SCARC_PROJECT,
            'token_fusion': 'wt_concat',
            'stage1_loss_mode': 'subcenter_arcface',
            'num_sub_centers': 4,
            'arcface_margin': 0.3,
            'arcface_scale': 64.0,
            'arcface_prune_warmup_epochs': 3,
            'arcface_prune_interval': 2,
            'arcface_prune_min_usage': 2,
            'arcface_prune_max_remove_per_class': 1,
        },
    },

    # ======================================================================
    # 비교용 예시 3: Stage-2 / Sub-center ArcFace 기반 fine-tuning
    # - Stage-1에서 학습한 Sub-center ArcFace checkpoint를 teacher로 사용
    # - Stage-2에서는 LSCE + optional KD
    # - suffix로 결과 폴더 분리
    # ======================================================================
    {
        'config': './config/SURFACE_ANODE_classification.yaml',
        'overrides': {
            'project_name': STAGE1_SCARC_PROJECT,
            'use_stage2': True,
            'token_fusion': 'wt_concat',
            'stage1_loss_mode': 'subcenter_arcface',
            'stage2_backbone_path': r'D:\NFF_ModelDeveloper\models\modi_dino\dino_small.bin',
            'stage2_max_epoch': 12,
            'stage2_label_smoothing': 0.05,
            'stage2_kd_alpha': 0.5,
            'stage2_kd_temperature': 3.0,
            'stage2_append_suffix': True,
            'stage2_project_suffix': '_S2_LAST1',
            'stage2_teacher_project_name': STAGE1_SCARC_PROJECT,
            'stage2_train_scope': 'last_blocks',
            'stage2_unfreeze_last_n_blocks': 1,
            'stage2_backbone_lr': 1e-6,
        },
    },

]


# ===========================================================================
# 평가만 실행하고 싶을 때 사용하세요 (SCHEDULE=[] 일 때만 동작)
# ===========================================================================
EVAL_ONLY_SCHEDULE = [
    # ----- ANODE: sum -----
    {
        'config': './config/SURFACE_ANODE_classification.yaml',
        'overrides': {
            'project_name': f'SURFACE_ANODE_DiNO_Sum_0729',
            'token_fusion': 'sum',
        },
    },

]
# ===========================================================================


def run_one(config_path: str, overrides: dict, run_logger: logging.Logger):
    """단일 학습 실행"""

    args = argument()
    args.load(config_path)

    # overrides 적용
    for key, value in overrides.items():
        setattr(args, key, value)

    use_stage2 = bool(getattr(args, 'use_stage2', False))
    base_project_name = args.project_name

    # Stage-2 실험명 suffix 정책
    if use_stage2 and bool(getattr(args, 'stage2_append_suffix', False)):
        suffix = str(getattr(args, 'stage2_project_suffix', '_S2'))
        if suffix and not str(args.project_name).endswith(suffix):
            args.project_name = f'{args.project_name}{suffix}'

    DEVICE = torch.device(args.device)

    train_logger = setup_logger(
        f'train-{args.project_name}',
        log_file=f'{args.save_dir}/{args.project_name}/train.log'
    )
    train_logger.info(f'Start training: {args.project_name}')
    train_logger.info(f'Config: {config_path}')
    train_logger.info(f'Overrides: {overrides}')
    train_logger.info(f'Full args: {args.__dict__}')

    model = get_models.build_model(args)
    head_type = getattr(args, 'head_type', 'transformer')
    train_logger.info(f'Model head type: {head_type}')
    train_logger.info(f'Pretrained model: {args.pretrained}')
    train_logger.info(f'Use stage2: {use_stage2}')
    print(f'**** Model head type : {head_type} ****')
    print(f'**** Pretrained model : {args.pretrained} ****')

    if args.finetuning:
        model = get_models.get_finetune_model(model, args)
        train_logger.info(f'Fine tuning: {args.finetuning_weight}')
        print(f'**** fine tunning model : {args.project_name}, {args.finetuning_weight} ****\n')

    # Stage-2: modi_dino 백본 로드 + stage2 epoch 반영
    if use_stage2:
        # suffix 사용 시에도 teacher를 stage1 프로젝트 폴더에서 자동 탐색 가능하도록 보정
        teacher_weight = getattr(args, 'stage2_teacher_weight', '')
        teacher_project_name = getattr(args, 'stage2_teacher_project_name', base_project_name)
        if not teacher_weight:
            teacher_root = os.path.join(args.save_dir, teacher_project_name)
            teacher_candidates = glob.glob(os.path.join(teacher_root, 'f1_score*_Loss*_epoch*.pth'))
            if teacher_candidates:
                def _extract_f1(path_str):
                    try:
                        return float(os.path.basename(path_str).split('f1_score')[1].split('_Loss')[0])
                    except Exception:
                        return -1.0

                best_teacher = max(teacher_candidates, key=_extract_f1)
                args.stage2_teacher_weight = best_teacher
                train_logger.info(f'[Stage2] Teacher auto-selected: {best_teacher}')

        stage2_backbone_path = getattr(args, 'stage2_backbone_path', '')
        if stage2_backbone_path and os.path.exists(stage2_backbone_path):
            stage2_state = torch.load(stage2_backbone_path, map_location='cpu')
            if hasattr(model, 'backbone'):
                model.backbone.load_state_dict(stage2_state, strict=False)
            else:
                model.load_state_dict(stage2_state, strict=False)
            train_logger.info(f'[Stage2] Fine-tuned backbone loaded: {stage2_backbone_path}')
        else:
            train_logger.warning(f'[Stage2] stage2_backbone_path not found: {stage2_backbone_path}')

        args.max_epoch = int(getattr(args, 'stage2_max_epoch', args.max_epoch))
        if hasattr(model, 'freeze_backbone'):
            model.freeze_backbone = False

    model.to(DEVICE)

    optimizer, sched = schedulers.get_optim_scheduler(args, model)
    train_logger.info(f'Optimizer and scheduler: {args.scheduler}')

    train_data, valid_data, train_labels, valid_labels = data_setup.get_trainList(args)
    train_dataset, valid_dataset = datasets.get_dataset(train_data, valid_data, train_labels, valid_labels, args)
    train_logger.info(f'Dataset loaded: train={len(train_data)}, valid={len(valid_data)}')

    if use_stage2:
        training.run_stage2_training(
            model, optimizer, sched,
            train_dataset, valid_dataset,
            args, logger=train_logger
        )
    else:
        training.run_training(
            model, optimizer, sched,
            train_dataset, valid_dataset,
            args, logger=train_logger
        )

    return args


def collect_test_data(args):
    """평가용 테스트 데이터 수집"""
    image_data = {
        label: data_setup.load_images_by_label(args.test_dir, label, mode='test')
        for label in args.labels
    }

    filenames_total = []
    label_total = []

    for label, images in image_data.items():
        filenames_total.extend(images)
        label_total.extend([label] * len(images))

    return filenames_total, label_total


def build_eval_dataloader(args, filenames_total, label_total, is_smw_eval: bool):
    """데이터셋 종류에 맞는 평가용 DataLoader 생성"""
    if is_smw_eval:
        return None

    ds_name = args.datasets_name.lower()
    if 'surface' in ds_name:
        dataset = datasets.SURFACEDataset(filenames_total, label_total, args, mode='valid')
    elif 'lhs' in ds_name:
        dataset = datasets.LHSDataset(filenames_total, label_total, args, mode='valid')
    elif 'ftf_folding' in ds_name:
        dataset = datasets.FoldingDataset(filenames_total, label_total, args, mode='valid')
    elif 'smw' in ds_name:
        dataset = None
    else:
        dataset = datasets.NormalDataset(filenames_total, label_total, args, mode='valid')

    if dataset is None:
        return None

    return DataLoader(
        dataset,
        batch_size=args.batch_size,
        num_workers=args.num_workers,
        shuffle=False,
        pin_memory=True,
        persistent_workers=(args.num_workers > 0)
    )


def get_cm_display_order(args, is_smw_eval: bool):
    """데이터셋별 confusion matrix 표시 순서"""
    if is_smw_eval:
        return ['NG_NO', 'NG_PINHOLE', 'NG_SPATTER', 'OK', 'OK_OVL']

    if 'surface' in args.datasets_name.lower():
        return [
            'CRACK_FOIL_1', 'CRACK_FOIL_2', 'CRACK_1', 'CRACK_2', 'DEBRIS',
            'PROTRUSION_1', 'PROTRUSION_2', 'PROTRUSION_3', 'CRATER_1', 'CRATER_2', 'SCRATCH_TINY'
        ]

    return args.labels


def run_eval(args, schedule_logger: logging.Logger):
    """학습 완료 모델 평가 실행"""
    eval_logger = setup_logger(
        f'eval-{args.project_name}',
        log_file=f'{args.save_dir}/{args.project_name}/eval.log'
    )

    filenames_total, label_total = collect_test_data(args)
    is_smw_eval = args.datasets_name.lower() == 'smw'
    dataloader = build_eval_dataloader(args, filenames_total, label_total, is_smw_eval)
    cm_display_order = get_cm_display_order(args, is_smw_eval)

    eval_manager = EvalManager(args, is_smw=is_smw_eval, logger=eval_logger)
    eval_results = eval_manager.run(
        filenames_total,
        label_total,
        dataloader=dataloader,
        cm_display_order=cm_display_order
    )

    schedule_logger.info(
        f"[EVAL SUCCESS] {args.project_name} | evaluated_models={len(eval_results)}"
    )

    return eval_results


def build_args(config_path: str, overrides: dict):
    """config + overrides 로 args 생성"""
    args = argument()
    args.load(config_path)

    for key, value in overrides.items():
        setattr(args, key, value)

    return args


def main_eval_only():
    """평가 전용 실행"""
    schedule_logger = setup_logger(
        'run-schedule',
        log_file=f'./run_schedule_{datetime.now().strftime("%Y%m%d_%H%M%S")}.log'
    )

    eval_total = len(EVAL_ONLY_SCHEDULE)
    schedule_logger.info(f'========== Eval-only mode started: {eval_total} jobs ==========')

    results = []

    if eval_total == 0:
        schedule_logger.info('EVAL_ONLY_SCHEDULE is empty. Nothing to run.')
        print('\n' + '='*60)
        print('  Eval-only mode: 실행할 작업이 없습니다.')
        print('  EVAL_ONLY_SCHEDULE를 채워주세요.')
        print('='*60)
        return

    for idx, job in enumerate(EVAL_ONLY_SCHEDULE, start=1):
        config_path = job['config']
        overrides = job.get('overrides', {})
        project_name = overrides.get('project_name', config_path)

        schedule_logger.info(f'\n---------- Eval Job {idx}/{eval_total}: {project_name} ----------')
        schedule_logger.info(f'Config: {config_path}  |  Overrides: {overrides}')
        print(f'\n{"="*60}')
        print(f'  Eval Job {idx}/{eval_total}: {project_name}')
        print(f'{"="*60}\n')

        start_time = time.time()
        try:
            args = build_args(config_path, overrides)
            run_eval(args, schedule_logger)
            elapsed = time.time() - start_time
            results.append({
                'job': project_name,
                'status': 'EVAL_SUCCESS',
                'elapsed': elapsed,
            })
            schedule_logger.info(f'[EVAL SUCCESS] {project_name} ({elapsed/3600:.2f}h)')
        except Exception as e:
            elapsed = time.time() - start_time
            results.append({
                'job': project_name,
                'status': 'EVAL_FAILED',
                'elapsed': elapsed,
                'error': str(e),
            })
            schedule_logger.error(f'[EVAL FAILED] {project_name} - {e}')
            schedule_logger.error(traceback.format_exc())

        finally:
            gc.collect()
            if torch.cuda.is_available():
                torch.cuda.empty_cache()

    schedule_logger.info('\n========== Eval-only mode finished ==========')
    print('\n' + '='*60)
    print('  Eval-only 완료 요약')
    print('='*60)
    for r in results:
        elapsed_str = f"{r['elapsed']/3600:.2f}h"
        if r['status'] == 'EVAL_SUCCESS':
            print(f"  [OK] {r['job']}  ({elapsed_str})")
        else:
            print(f"  [FAIL] {r['job']}  ({elapsed_str})  ->  {r.get('error','')}")
    print('='*60)

    for r in results:
        elapsed_str = f"{r['elapsed']/3600:.2f}h"
        status = 'SUCCESS' if r['status'] == 'EVAL_SUCCESS' else f"FAILED: {r.get('error','')}"
        schedule_logger.info(f"  {r['job']} | EVAL: {status}({elapsed_str})")


def main():
    schedule_logger = setup_logger(
        'run-schedule',
        log_file=f'./run_schedule_{datetime.now().strftime("%Y%m%d_%H%M%S")}.log'
    )

    total = len(SCHEDULE)
    schedule_logger.info(f'========== Schedule started: {total} jobs ==========')

    results = []

    for idx, job in enumerate(SCHEDULE, start=1):
        config_path = job['config']
        overrides = job.get('overrides', {})
        project_name = overrides.get('project_name', config_path)

        schedule_logger.info(f'\n---------- Job {idx}/{total}: {project_name} ----------')
        schedule_logger.info(f'Config: {config_path}  |  Overrides: {overrides}')
        print(f'\n{"="*60}')
        print(f'  Job {idx}/{total}: {project_name}')
        print(f'{"="*60}\n')

        start_time = time.time()
        try:
            args = run_one(config_path, overrides, schedule_logger)
            elapsed = time.time() - start_time
            msg = f'[TRAIN SUCCESS] {project_name} ({elapsed/3600:.2f}h)'
            schedule_logger.info(msg)

            results.append({
                'job': project_name,
                'status': 'SUCCESS',
                'elapsed': elapsed,
            })

        except Exception as e:
            elapsed = time.time() - start_time
            msg = f'[FAILED]  {project_name} - {e}'
            schedule_logger.error(msg)
            schedule_logger.error(traceback.format_exc())
            results.append({
                'job': project_name,
                'status': 'FAILED',
                'error': str(e),
                'elapsed': elapsed,
            })
            print(f'[ERROR] {e}')
            print('다음 작업으로 넘어갑니다...\n')

        finally:
            # GPU 메모리 정리
            gc.collect()
            if torch.cuda.is_available():
                torch.cuda.empty_cache()

    # 최종 요약
    schedule_logger.info('\n========== Schedule finished ==========')
    print('\n' + '='*60)
    print('  Training Schedule 완료 요약')
    print('='*60)
    for r in results:
        elapsed_str = f"{r['elapsed']/3600:.2f}h"
        if r['status'] == 'SUCCESS':
            print(f"  [OK] {r['job']}  ({elapsed_str})")
        else:
            print(f"  [FAIL] {r['job']}  ({elapsed_str})  ->  {r.get('error','')}")
    print('='*60)

    for r in results:
        elapsed_str = f"{r['elapsed']/3600:.2f}h"
        if r['status'] == 'SUCCESS':
            status = f"TRAIN: SUCCESS({elapsed_str})"
        else:
            status = f"TRAIN: FAILED({elapsed_str}) - {r.get('error','')}"

        schedule_logger.info(f"  {r['job']} | {status}")


if __name__ == '__main__':
    main()
    # main_eval_only()
