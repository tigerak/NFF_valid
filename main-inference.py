import os

# Pytorch Imports
os.environ['KMP_DUPLICATE_LIB_OK'] = 'True'

# user library 
from argments import argument
from evaluation.inference import InferenceManager, setup_logger


# ============================================================================
# 사용 예제
# ============================================================================
"""
# 표준 모델 추론 (폴더)
config_path = './config/SURFACE_ANODE_classification.yaml'
path_list = "D:\\Images\\test_images"
args = argument()
args.load(config_path)
manager = InferenceManager(args, is_smw=False)
manager.run(path_list)

# SMW 모델 추론 (폴더)
config_path = './config/SMW_classification.yaml'
path_list = "D:\\Images\\SMW_test_images"
args = argument()
args.load(config_path)
manager = InferenceManager(args, is_smw=True)
manager.run(path_list)

# 이미지 리스트로 추론
config_path = './config/SURFACE_ANODE_classification.yaml'
image_list = [
    "D:\\Images\\image1.bmp",
    "D:\\Images\\image2.bmp",
]
args = argument()
args.load(config_path)
manager = InferenceManager(args, is_smw=False)
manager.run(image_list)
"""

if __name__ == '__main__':
    # ========================================================================
    # 여기에 실제 사용할 경로와 설정을 입력하세요
    # ========================================================================
    
    # 설정 파일 경로
    config_path = './config/SURFACE_ANODE_classification.yaml'  # 수정 필요
    
    # 이미지 경로 (폴더, 단일 파일, 또는 리스트)
    path_list = r"D:\NFF\00.WINDER_SURFACE\test_3000"
    
    # SMW 모델 여부
    is_smw_eval = False  # 수정 필요
    
    # ========================================================================
    # 추론 실행
    # ========================================================================
    args = argument()
    args.load(config_path)
    
    # 로거 설정 (선택사항)
    logger = setup_logger('main-inference', log_file='./inference.log')
    
    # InferenceManager 초기화 및 실행
    manager = InferenceManager(args, is_smw=is_smw_eval, logger=logger)
    results = manager.run(path_list)
