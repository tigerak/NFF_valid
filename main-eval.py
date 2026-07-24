import os

# For data manipulation
import pandas as pd

# Pytorch Imports
os.environ['KMP_DUPLICATE_LIB_OK'] = 'True'

from torch.utils.data import DataLoader
import torch.nn as nn

# user library 
from argments import argument
from utils import get_models, datasets
from dataset import data_setup

from evaluation.eval import EvalManager, setup_logger
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

def main():
    """메인 평가 함수"""
    # 설정 파일 경로 (필요시 변경)
    config_path = './config/SURFACE_ANODE_classification.yaml'
    # config_path = './config/SURFACE_CATHODE_classification.yaml'
    # config_path = './config/SMW_classification.yaml'

    # 설정 로드
    args = argument()
    args.load(config_path)
    
    # 로거 설정
    logger = setup_logger('main-eval', log_file=f'{args.save_dir}/{args.project_name}/eval.log')
    logger.info(f'Pretrained model : {args.pretrained}')

    # 테스트 데이터 수집
    filenames_total, label_total = collect_test_data(args)
    is_smw_eval = args.datasets_name.lower() == 'smw'

    # 표준 모델용 데이터로더 준비 (표준 모델의 경우에만)
    dataloader = None
    if not is_smw_eval:
        dataset = datasets.SURFACEDataset(filenames_total, label_total, args, mode='valid')
        dataloader = DataLoader(
            dataset,
            batch_size=args.batch_size,
            num_workers=args.num_workers,
            shuffle=False,
            pin_memory=True,
            persistent_workers=(args.num_workers > 0)
        )

    # 평가 설정
    cm_display_order = args.labels
    # if is_smw_eval:
    #     cm_display_order = ['NG_NO',  'NG_PINHOLE', 'NG_SPATTER', 'OK', "OK_OVL"]
    # else:
    #     cm_display_order = [
    #         'CRACK_FOIL_1','CRACK_FOIL_2', 'CRACK_1', 'CRACK_2', 'DEBRIS',
    #         'PROTRUSION_1', 'PROTRUSION_2', 'PROTRUSION_3', 'CRATER_1', 'CRATER_2', 'SCRATCH_TINY'
    #     ]

    # 평가 매니저 생성 및 실행
    eval_manager = EvalManager(args, is_smw=is_smw_eval, logger=logger)
    eval_results = eval_manager.run(
        filenames_total,
        label_total,
        dataloader=dataloader,
        cm_display_order=cm_display_order
    )

    logger.info(f"Evaluation completed with {len(eval_results)} model(s)")

if __name__ == '__main__':
    main()