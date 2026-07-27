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
    DEVICE = torch.device(args.device)

    logger = setup_logger('main-train', log_file=f'{args.save_dir}/{args.project_name}/train.log')
    logger.info(f"Start training: {args.project_name}")
    logger.info(f"Config: {args.__dict__}")

    ## get models (configurable head: args.head_type)
    model = get_models.build_model(args)
    head_type = getattr(args, 'head_type', 'transformer')
    print(f'**** Model head type : {head_type} ****')
    print(f'**** Pretrained model : {args.pretrained} ****')
    
    ## if finetunning .. 
    if args.finetuning != False : 
        model = get_models.get_finetune_model(model, args)
        print(f'**** fine tunning model : {args.project_name}, {args.finetuning_weight} ****\n')

    model.to(DEVICE)

    ## get scheduler
    optimizer, scheduler = schedulers.get_optim_scheduler(args, model)

    ## get dataset
    train_data, valid_data, train_labels, valid_labels = data_setup.get_trainList(args)
    train_dataset, valid_dataset = datasets.get_dataset(train_data, valid_data, train_labels, valid_labels, args)

    ## Run training 
    training.run_training(model, 
                          optimizer, 
                          scheduler,
                          train_dataset,
                          valid_dataset,
                          args)








