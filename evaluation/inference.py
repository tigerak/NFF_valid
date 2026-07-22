"""
추론 엔진 모듈

표준 모델과 SMW 모델의 대량 추론을 처리하며, 결과를 클래스별로 분류하여 저장합니다.
"""

import os
import glob
import shutil

import pandas as pd
import numpy as np
import torch
from tqdm import tqdm

from evaluation.base_engine import (
    BaseEngine,
    ImageLoader,
    ResultsSaver,
    load_model,
    export_onnx_model,
    setup_logger,
)
from evaluation.smw_eval import SMW_extract_patches, SMW_get_result, SMW_get_pred_image


# ============================================================================
# 표준 모델 추론 엔진
# ============================================================================
class StandardInferenceEngine(BaseEngine):
    """표준 분류 모델 추론 엔진"""
    
    def run(self, model, image_paths, model_save_dir):
        """표준 모델 추론
        
        Args:
            model: 학습된 모델
            image_paths: 이미지 경로 리스트
            model_save_dir: 모델 결과 저장 디렉토리
        
        Returns:
            results_df: 결과 데이터프레임
        """
        saver = ResultsSaver(model_save_dir, self.args.labels, self.logger)
        saver.cleanup_results_directory()
        saver.create_class_directories()
        
        all_predictions = []
        all_filenames = []
        all_confidences = []
        
        with torch.no_grad():
            for image_path in tqdm(image_paths, desc='Standard Model Inference', ascii=True):
                if not os.path.exists(image_path):
                    self.logger.warning(f"Image not found: {image_path}")
                    continue
                
                try:
                    # 이미지 처리
                    img_tensor = self.load_and_preprocess_image(image_path)
                    img_tensor = img_tensor.unsqueeze(0).to(self.device)
                    
                    # 추론
                    outputs = model(img_tensor)
                    probs = torch.softmax(outputs, dim=1)
                    pred_label = torch.argmax(outputs, dim=1).item()
                    confidences = probs.cpu().numpy()[0]
                    
                    # 결과 저장
                    all_predictions.append(pred_label)
                    all_filenames.append(image_path)
                    all_confidences.append(confidences)
                    
                    # 이미지 복사
                    pred_class_name = self.args.labels[pred_label]
                    saver.save_image_to_class(image_path, pred_class_name)
                    
                except Exception as e:
                    self.logger.error(f"Error processing {image_path}: {e}")
                    continue
        
        # 결과 데이터프레임 생성
        results_df = pd.DataFrame({
            'image_name': [os.path.basename(f) for f in all_filenames],
            'image_path': all_filenames,
            'pred_class': [self.args.labels[pred] for pred in all_predictions],
        })
        
        # 모든 클래스의 confidence 추가
        confidence_df = pd.DataFrame(
            all_confidences,
            columns=[f'confidence_{cls}' for cls in self.args.labels]
        )
        results_df = pd.concat([results_df, confidence_df], axis=1)
        
        return results_df


# ============================================================================
# SMW 모델 추론 엔진
# ============================================================================
class SMWInferenceEngine(BaseEngine):
    """SMW 모델 추론 엔진 (56개 패치 분리 + 후처리)"""
    
    def run(self, model, image_paths, model_save_dir):
        """SMW 모델 추론
        
        Args:
            model: 학습된 모델
            image_paths: 이미지 경로 리스트
            model_save_dir: 모델 결과 저장 디렉토리
        
        Returns:
            results_df: 결과 데이터프레임
        """
        saver = ResultsSaver(model_save_dir, self.args.labels[:-1], self.logger)  # OK_OVERLAP 제외
        saver.cleanup_results_directory()
        saver.create_class_directories(exclude_labels=['OK_OVERLAP'])
        
        all_predictions = []
        all_filenames = []
        all_patch_indices = []
        all_patch_confidences = []
        
        with torch.no_grad():
            for image_path in tqdm(image_paths, desc='SMW Model Inference', ascii=True):
                if not os.path.exists(image_path):
                    self.logger.warning(f"Image not found: {image_path}")
                    continue
                
                try:
                    # 패치 추출
                    origins, patches, positions = SMW_extract_patches(
                        image_path,
                        patch_width=self.args.img_size[0],
                        stride=self.args.img_size[0],
                    )
                    patches = patches.to(self.device, dtype=torch.float32)
                    
                    # 패치별 추론
                    outputs = model(patches)
                    confidences = torch.softmax(outputs, dim=1)
                    patch_preds = torch.argmax(outputs, dim=1).cpu().numpy()
                    patch_confidences = confidences.cpu().numpy()
                    
                    # 최종 판정 (후처리)
                    final_pred, final_score = SMW_get_result(patch_preds.copy())
                    
                    # 불량 패치 정보 추출
                    defect_patches = np.where(patch_preds != 5)[0]  # 5 = OK
                    
                    if len(defect_patches) > 0:
                        patch_idx = defect_patches[0]
                        patch_conf = np.max(patch_confidences[patch_idx])
                    else:
                        patch_idx = 0
                        patch_conf = np.max(patch_confidences[0])
                    
                    all_predictions.append(final_pred)
                    all_filenames.append(image_path)
                    all_patch_indices.append(patch_idx)
                    all_patch_confidences.append(patch_conf)
                    
                    # 이미지 복사
                    pred_class_name = self.args.labels[final_pred]
                    if pred_class_name == 'OK_OVERLAP':
                        pred_class_name = 'OK'
                    saver.save_image_to_class(image_path, pred_class_name)
                    
                    # Overlay 이미지 생성 (불량만: pred_class ≠ 5,6)
                    if final_pred not in [5, 6]:
                        self._save_overlay_image(
                            patch_preds, image_path, pred_class_name,
                            os.path.join(model_save_dir, 'results', pred_class_name)
                        )
                    
                except Exception as e:
                    self.logger.error(f"Error processing {image_path}: {e}")
                    continue
        
        # 결과 데이터프레임 생성
        results_df = pd.DataFrame({
            'image_name': [os.path.basename(f) for f in all_filenames],
            'image_path': all_filenames,
            'pred_class': [self.args.labels[pred] for pred in all_predictions],
            'patch_index': all_patch_indices,
            'patch_confidence_max': all_patch_confidences,
        })
        
        return results_df
    
    def _save_overlay_image(self, patch_preds, image_path, pred_class_name, save_dir):
        """Overlay 이미지 저장
        
        Args:
            patch_preds: 패치별 예측 결과
            image_path: 원본 이미지 경로
            pred_class_name: 예측 클래스명
            save_dir: 저장 디렉토리
        """
        try:
            image_stem = os.path.splitext(os.path.basename(image_path))[0]
            overlay_path = os.path.join(save_dir, f"{image_stem}_overlay.png")
            
            SMW_get_pred_image(
                pred=patch_preds,
                img=image_path,
                label=pred_class_name,
                save_path=overlay_path,
                show=False,
            )
        except Exception as e:
            self.logger.warning(f"Failed to save overlay image for {image_path}: {e}")


# ============================================================================
# 추론 매니저 (팩토리 패턴)
# ============================================================================
class InferenceManager:
    """추론 엔진 관리 및 조율 클래스"""
    
    def __init__(self, args, is_smw=False, logger=None):
        """
        Args:
            args: 설정 객체
            is_smw: SMW 모델 여부
            logger: 로거 객체
        """
        self.args = args
        self.is_smw = is_smw
        self.logger = logger or setup_logger(__name__)
        
        # 추론 엔진 선택
        if is_smw:
            self.engine = SMWInferenceEngine(args, self.logger)
        else:
            self.engine = StandardInferenceEngine(args, self.logger)
    
    def run(self, image_paths):
        """추론 실행
        
        Args:
            image_paths: 이미지 경로 리스트 또는 폴더 경로
        
        Returns:
            inference_results: 모델별 추론 결과 딕셔너리
        """
        # 이미지 경로 수집
        image_paths = ImageLoader.load_image_paths(image_paths, recursive=True)
        
        self.logger.info(f"Dataset: {self.args.datasets_name}")
        self.logger.info(f"Total images to process: {len(image_paths)}")
        
        if len(image_paths) == 0:
            self.logger.error("No images found!")
            return {}
        
        # 모델 폴더의 모든 .pth 파일 순회
        model_dir = os.path.join(self.args.save_dir, self.args.project_name)
        model_files = sorted(glob.glob(os.path.join(model_dir, "*.pth")))
        
        if len(model_files) == 0:
            self.logger.error(f"No model files found in {model_dir}")
            return {}
        
        self.logger.info(f"Found {len(model_files)} model(s) to evaluate")
        
        # 모델별 추론
        inference_results = {}
        
        for model_path in model_files:
            model_name = os.path.splitext(os.path.basename(model_path))[0]
            model_save_dir = os.path.join(model_dir, 'inference_results', model_name)
            
            self.logger.info(f"\n{'='*60}")
            self.logger.info(f"Evaluating model: {model_name}")
            self.logger.info(f"Save directory: {model_save_dir}")
            self.logger.info(f"{'='*60}")
            
            try:
                # 모델 로드
                device = torch.device(self.args.device)
                model = load_model(self.args, model_path, device)
                
                # 추론 실행
                results_df = self.engine.run(model, image_paths, model_save_dir)
                
                # 결과 통계 출력
                self.logger.info(f"\nInference Results:")
                self.logger.info(f"\n{results_df['pred_class'].value_counts()}")
                
                # 결과 저장
                saver = ResultsSaver(model_save_dir, self.args.labels, self.logger)
                csv_name = f'{model_name}_inference_results.csv'
                saver.save_results_csv(results_df, csv_name)
                
                inference_results[model_name] = {
                    'results_df': results_df,
                    'model_path': model_path,
                    'model_save_dir': model_save_dir,
                }
                
            except Exception as e:
                self.logger.error(f"Error evaluating model {model_name}: {e}")
                import traceback
                traceback.print_exc()
                continue
        
        self.logger.info(f"\n{'='*60}")
        self.logger.info(f"Inference completed! Processed {len(inference_results)} model(s)")
        self.logger.info(f"{'='*60}")
        
        return inference_results
