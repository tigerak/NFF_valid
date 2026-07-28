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
"""

import os
import gc
import time
import traceback
import logging
from datetime import datetime

import torch

from argments import argument
from utils import get_models, datasets, schedulers, training
from dataset import data_setup
from evaluation.base_engine import setup_logger

# ===========================================================================
# 여기에 실행할 작업 목록을 정의하세요
# ===========================================================================
TODAY = datetime.now().strftime('%m%d')

SCHEDULE = [
    # # ----- ANODE: sum -----
    # {
    #     'config': './config/SURFACE_ANODE_classification.yaml',
    #     'overrides': {
    #         'project_name': f'SURFACE_ANODE_DiNO_Sum_{TODAY}',
    #         'token_fusion': 'sum',
    #     },
    # },

    # ----- ANODE: weight concat -----
    {
        'config': './config/SURFACE_ANODE_classification.yaml',
        'overrides': {
            'project_name': f'SURFACE_ANODE_DiNO_Concat_{TODAY}',
            'token_fusion': 'wt_concat',
            'focal_alpha': [1., 1., 1., 1., 1., 1., 1., 1., 1., 1., 1.]
        },
    },

    # ----- ANODE: Attention / focal alpha 1. -----
    {
        'config': './config/SURFACE_ANODE_classification.yaml',
        'overrides': {
            'project_name': f'SURFACE_ANODE_DiNO_Attn_{TODAY}',
            'token_fusion': 'attn',
            'focal_alpha': [1., 1., 1., 1., 1., 1., 1., 1., 1., 1., 1.]
        },
    },

    # ----- ANODE: Attention / focal alpha .5 -----
    {
        'config': './config/SURFACE_ANODE_classification.yaml',
        'overrides': {
            'project_name': f'SURFACE_ANODE_DiNO_Attn_FA05_{TODAY}',
            'token_fusion': 'attn',
            'focal_alpha': [.5, .5, .5, .5, .5, .5, .5, .5, .5, .5, .5]
        },
    },

    
    # ----- ANODE: Attention / focal alpha 1. / Mixup -----
    {
        'config': './config/SURFACE_ANODE_classification.yaml',
        'overrides': {
            'project_name': f'SURFACE_ANODE_DiNO_Attn_FA1_Mixup_{TODAY}',
            'token_fusion': 'attn',
            'focal_alpha': [1., 1., 1., 1., 1., 1., 1., 1., 1., 1., 1.],
            'use_mixup': True,
            'mixup_alpha': 1.0
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
    print(f'**** Model head type : {head_type} ****')
    print(f'**** Pretrained model : {args.pretrained} ****')

    if args.finetuning:
        model = get_models.get_finetune_model(model, args)
        train_logger.info(f'Fine tuning: {args.finetuning_weight}')
        print(f'**** fine tunning model : {args.project_name}, {args.finetuning_weight} ****\n')

    model.to(DEVICE)

    optimizer, sched = schedulers.get_optim_scheduler(args, model)
    train_logger.info(f'Optimizer and scheduler: {args.scheduler}')

    train_data, valid_data, train_labels, valid_labels = data_setup.get_trainList(args)
    train_dataset, valid_dataset = datasets.get_dataset(train_data, valid_data, train_labels, valid_labels, args)
    train_logger.info(f'Dataset loaded: train={len(train_data)}, valid={len(valid_data)}')

    training.run_training(
        model, optimizer, sched,
        train_dataset, valid_dataset,
        args, logger=train_logger
    )


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
            run_one(config_path, overrides, schedule_logger)
            elapsed = time.time() - start_time
            msg = f'[SUCCESS] {project_name} ({elapsed/3600:.2f}h)'
            schedule_logger.info(msg)
            results.append({'job': project_name, 'status': 'SUCCESS', 'elapsed': elapsed})

        except Exception as e:
            elapsed = time.time() - start_time
            msg = f'[FAILED]  {project_name} — {e}'
            schedule_logger.error(msg)
            schedule_logger.error(traceback.format_exc())
            results.append({'job': project_name, 'status': 'FAILED', 'error': str(e), 'elapsed': elapsed})
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
            print(f"  ✅ {r['job']}  ({elapsed_str})")
        else:
            print(f"  ❌ {r['job']}  ({elapsed_str})  →  {r.get('error','')}")
    print('='*60)

    for r in results:
        elapsed_str = f"{r['elapsed']/3600:.2f}h"
        status = 'SUCCESS' if r['status'] == 'SUCCESS' else f"FAILED: {r.get('error','')}"
        schedule_logger.info(f"  {r['job']} | {status} | {elapsed_str}")


if __name__ == '__main__':
    main()
