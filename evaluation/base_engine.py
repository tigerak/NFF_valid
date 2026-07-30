"""
추론/평가 엔진 기본 모듈

공통 로직, 유틸리티, 기반 클래스를 제공합니다.
"""

import os
import logging
from abc import ABC, abstractmethod

import pandas as pd
import numpy as np
import torch
from torchvision import transforms
from PIL import Image
from tqdm import tqdm

from utils import get_models
from utils import losses


# ============================================================================
# 로깅 설정
# ============================================================================
def setup_logger(name, log_file=None):
    """로거 설정
    
    Args:
        name: 로거 이름
        log_file: 로그 파일 경로 (선택사항)
    
    Returns:
        logger: 설정된 로거 객체
    """
    logger = logging.getLogger(name)
    logger.setLevel(logging.INFO)
    
    formatter = logging.Formatter(
        '%(asctime)s - %(name)s - %(levelname)s - %(message)s'
    )
    
    # 콘솔 핸들러
    console_handler = logging.StreamHandler()
    console_handler.setFormatter(formatter)
    logger.addHandler(console_handler)
    
    # 파일 핸들러 (선택사항)
    if log_file:
        os.makedirs(os.path.dirname(log_file) or '.', exist_ok=True)
        file_handler = logging.FileHandler(log_file)
        file_handler.setFormatter(formatter)
        logger.addHandler(file_handler)
    
    return logger


# ============================================================================
# 이미지 로더
# ============================================================================
class ImageLoader:
    """이미지 경로 수집 및 관리 클래스"""
    
    DEFAULT_EXTENSIONS = ['*.bmp', '*.BMP', '*.jpg', '*.JPG', '*.jpeg', '*.JPEG', '*.png', '*.PNG']
    
    @staticmethod
    def load_image_paths(path_list, recursive=False):
        """이미지 경로 수집
        
        Args:
            path_list: 폴더 경로, 단일 이미지 경로, 또는 경로 리스트
            recursive: 하위 폴더 재귀 탐색 여부
        
        Returns:
            image_paths: 이미지 파일 경로 리스트
        """
        import glob
        
        image_paths = []
        
        # 리스트인 경우
        if isinstance(path_list, (list, tuple)):
            for item in path_list:
                if os.path.isdir(item):
                    image_paths.extend(ImageLoader._load_from_directory(item, recursive))
                elif os.path.isfile(item):
                    image_paths.append(item)
        # 문자열인 경우
        elif isinstance(path_list, str):
            if os.path.isdir(path_list):
                image_paths = ImageLoader._load_from_directory(path_list, recursive)
            elif os.path.isfile(path_list):
                image_paths = [path_list]
        
        return list(set(image_paths))  # 중복 제거
    
    @staticmethod
    def _load_from_directory(directory, recursive=False):
        """디렉토리에서 이미지 로드
        
        Args:
            directory: 디렉토리 경로
            recursive: 재귀 탐색 여부
        
        Returns:
            image_paths: 이미지 파일 경로 리스트
        """
        import glob
        
        image_paths = []
        
        for ext in ImageLoader.DEFAULT_EXTENSIONS:
            if recursive:
                image_paths.extend(glob.glob(os.path.join(directory, '**', ext), recursive=True))
            else:
                image_paths.extend(glob.glob(os.path.join(directory, ext)))
        
        return image_paths


# ============================================================================
# 결과 저장 클래스
# ============================================================================
class ResultsSaver:
    """추론/평가 결과 저장 관리 클래스"""
    
    def __init__(self, save_dir, labels, logger=None):
        """
        Args:
            save_dir: 결과 저장 루트 디렉토리
            labels: 클래스 레이블 리스트
            logger: 로거 객체
        """
        self.save_dir = save_dir
        self.labels = labels
        self.logger = logger or setup_logger(__name__)
    
    def create_class_directories(self, exclude_labels=None):
        """클래스별 결과 디렉토리 생성
        
        Args:
            exclude_labels: 제외할 레이블 리스트 (예: ['OK_OVERLAP'])
        
        Returns:
            class_dirs: 생성된 클래스 디렉토리 딕셔너리
        """
        import shutil
        
        exclude_labels = exclude_labels or []
        class_dirs = {}
        
        for label in self.labels:
            if label not in exclude_labels:
                class_dir = os.path.join(self.save_dir, 'results', label)
                os.makedirs(class_dir, exist_ok=True)
                class_dirs[label] = class_dir
        
        return class_dirs
    
    def save_image_to_class(self, image_path, pred_class):
        """이미지를 해당 클래스 폴더에 복사
        
        Args:
            image_path: 원본 이미지 경로
            pred_class: 예측 클래스명
        """
        import shutil
        
        try:
            image_name = os.path.basename(image_path)
            dest_path = os.path.join(self.save_dir, 'results', pred_class, image_name)
            shutil.copy2(image_path, dest_path)
        except Exception as e:
            self.logger.warning(f"Failed to save image {image_path}: {e}")
    
    def save_results_csv(self, results_df, csv_name):
        """결과를 CSV로 저장
        
        Args:
            results_df: 결과 데이터프레임
            csv_name: CSV 파일명
        
        Returns:
            csv_path: 저장된 CSV 파일 경로
        """
        csv_path = os.path.join(self.save_dir, csv_name)
        results_df.to_csv(csv_path, index=False, encoding='utf-8-sig')
        self.logger.info(f"Results saved to: {csv_path}")
        return csv_path
    
    def cleanup_results_directory(self):
        """기존 결과 디렉토리 삭제"""
        import shutil
        
        results_dir = os.path.join(self.save_dir, 'results')
        if os.path.exists(results_dir):
            shutil.rmtree(results_dir)
            self.logger.info(f"Cleaned up: {results_dir}")


# ============================================================================
# 기본 추론/평가 엔진
# ============================================================================
class BaseEngine(ABC):
    """추론/평가 엔진 기본 클래스"""
    
    def __init__(self, args, logger=None):
        """
        Args:
            args: 설정 객체
            logger: 로거 객체
        """
        self.args = args
        self.device = torch.device(args.device)
        self.logger = logger or setup_logger(__name__)
    
    def load_and_preprocess_image(self, image_path):
        """이미지 로드 및 전처리
        
        Args:
            image_path: 이미지 경로
        
        Returns:
            img_tensor: 전처리된 이미지 텐서
        """
        transform = transforms.Compose([
            transforms.Resize(self.args.img_size),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.5, 0.5, 0.5], std=[1.0, 1.0, 1.0])
        ])
        
        img = Image.open(image_path).convert('RGB')
        img_tensor = transform(img)
        
        return img_tensor
    
    @abstractmethod
    def run(self, model, *args, **kwargs):
        """실행 (서브클래스에서 구현)"""
        pass


# ============================================================================
# 모델 로더
# ============================================================================
def load_model(args, weight_path, device, strict=True):
    """모델 로드
    
    Args:
        args: 설정 객체
        weight_path: 모델 가중치 경로
        device: 디바이스
    
    Returns:
        model: 로드된 모델
    """
    model = get_models.build_model(args)
    
    # 체크포인트는 두 형식을 모두 허용한다.
    # 1) 구형: raw model.state_dict
    # 2) 신형: {'model_state_dict': ..., 'arcface_state_dict': ..., ...}
    weight = torch.load(weight_path, map_location=device)
    model_state = losses.extract_model_state_dict(weight)
    model.load_state_dict(model_state, strict=strict)
    model.to(device)
    model.eval()
    
    return model


# ============================================================================
# ONNX 내보내기
# ============================================================================
def export_onnx_model(model, args, weight_path, batch_size):
    """ONNX 모델 내보내기
    
    Args:
        model: PyTorch 모델
        args: 설정 객체
        weight_path: 가중치 파일 경로
        batch_size: 배치 크기
    """
    onnx_path = os.path.join(
        args.save_dir, 
        args.project_name, 
        os.path.basename(weight_path).replace('.pth', '.onnx')
    )
    dummy_input = torch.randn(batch_size, 3, args.img_size[0], args.img_size[1]).to(args.device)

    torch.onnx.export(
        model,
        dummy_input,
        onnx_path,
        opset_version=14,
        input_names=['input'],
        output_names=['output'],
        dynamic_axes={
            'input': {0: 'batch_size'},
            'output': {0: 'batch_size'},
        }
    )
    logging.getLogger(__name__).info(f"ONNX exported: {onnx_path}")
