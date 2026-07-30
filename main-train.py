import os, time, glob, copy, gc
import torch.nn as nn
# For data manipulation
import numpy as np

# Pytorch Imports
##os.environ['KMP_DUPLICATE_LIB_OK']='True'
import torch 

## user library 
from argments import argument
from utils import get_models, datasets, schedulers, training
from dataset import data_setup
from evaluation.base_engine import setup_logger

import warnings
warnings.filterwarnings("ignore")

if __name__ == '__main__':

    ## Load Argument
    # config_path = './config/SURFACE_CATHODE_classification.yaml'
    config_path = './config/SURFACE_ANODE_classification.yaml'
    # config_path = './config/SMW_classification.yaml'

    args = argument()
    args.load(config_path)

    # Stage-2 실험명 구분(체크포인트/로그 덮어쓰기 방지)
    # 동일 project_name을 쓰면 Stage-1 결과를 덮어쓸 수 있으므로 suffix로 분리한다.
    _use_stage2 = bool(getattr(args, 'use_stage2', False))
    if _use_stage2 and bool(getattr(args, 'stage2_append_suffix', True)):
        suffix = str(getattr(args, 'stage2_project_suffix', '_S2_LAST1'))
        if suffix and not str(args.project_name).endswith(suffix):
            args.project_name = f"{args.project_name}{suffix}"

    DEVICE = torch.device(args.device)

    logger = setup_logger('main-train', log_file=f'{args.save_dir}/{args.project_name}/train.log')
    logger.info(f"Start training: {args.project_name}")
    logger.info(f"Config: {args.__dict__}")

    ## get models (configurable head: args.head_type)
    model = get_models.build_model(args)
    head_type = getattr(args, 'head_type', 'transformer')
    logger.info(f'Model head type: {head_type}')
    logger.info(f'Pretrained model: {args.pretrained}')
    print(f'**** Model head type : {head_type} ****')
    print(f'**** Pretrained model : {args.pretrained} ****')
    
    ## if finetunning ..
    if args.finetuning != False :
        model = get_models.get_finetune_model(model, args)
        logger.info(f'Fine tuning model: {args.project_name}, {args.finetuning_weight}')
        print(f'**** fine tunning model : {args.project_name}, {args.finetuning_weight} ****\n')

    ## ── Stage-2: modi_dino 파인튜닝 백본으로 교체 ─────────────────────────
    # 여기서는 "초기 backbone 상태"만 로드한다.
    # KD teacher 로드는 training.run_stage2_training 내부에서 별도로 수행된다.
    if _use_stage2:
        _s2_path = getattr(args, 'stage2_backbone_path', '')
        if _s2_path and os.path.exists(_s2_path):
            _s2_state = torch.load(_s2_path, map_location='cpu')
            model.backbone.load_state_dict(_s2_state, strict=False)
            logger.info(f'[Stage2] Fine-tuned backbone loaded: {_s2_path}')
        else:
            logger.warning(f'[Stage2] stage2_backbone_path not found: {_s2_path}')
        args.max_epoch = int(getattr(args, 'stage2_max_epoch', 12))
        # Stage-2에서는 backbone에 gradient가 흐르도록 freeze 동작 해제.
        # 실제로 어떤 블록을 학습할지는 run_stage2_training의
        # _set_stage2_trainable_blocks에서 결정한다.
        if hasattr(model, 'freeze_backbone'):
            model.freeze_backbone = False
    ## ──────────────────────────────────────────────────────────────────────

    model.to(DEVICE)

    ## get scheduler
    optimizer, scheduler = schedulers.get_optim_scheduler(args, model)
    logger.info(f'Optimizer and scheduler initialized: {args.scheduler}')

    ## get dataset
    train_data, valid_data, train_labels, valid_labels = data_setup.get_trainList(args)
    train_dataset, valid_dataset = datasets.get_dataset(train_data, valid_data, train_labels, valid_labels, args)
    logger.info(f'Dataset loaded: train={len(train_data)}, valid={len(valid_data)}')

    ## Run training
    if _use_stage2:
        logger.info('[Stage2] Running stage-2 (fine-tuned backbone + LSCE + optional KD)')
        training.run_stage2_training(model, optimizer, scheduler,
                                     train_dataset, valid_dataset,
                                     args, logger=logger)
    else:
        training.run_training(model, optimizer, scheduler,
                              train_dataset, valid_dataset,
                              args, logger=logger)








