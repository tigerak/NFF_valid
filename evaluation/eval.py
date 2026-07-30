"""
평가 엔진 모듈

테스트 셋에 대한 모델 평가 및 Confusion Matrix 생성
"""

import os
import glob
import logging

import pandas as pd
import numpy as np
import torch
from tqdm import tqdm
import torch.nn.functional as F

from evaluation.base_engine import (
    BaseEngine,
    ResultsSaver,
    load_model,
    export_onnx_model,
    setup_logger,
)
from utils import losses
from evaluation.smw_eval import SMW_extract_patches, SMW_get_result, SMW_get_pred_image
from evaluation.get_result_summary import (
    build_results_dataframe,
    normalize_smw_label,
    summarize_classification_results,
    save_results_dataframe,
)


# ============================================================================
# 표준 평가 엔진
# ============================================================================
class StandardEvalEngine(BaseEngine):
    """테스트 데이터셋에 대한 표준 모델 평가"""
    
    def run(self, model, dataloader):
        """표준 모델 평가
        
        Args:
            model: 학습된 모델
            dataloader: 테스트 데이터로더
        
        Returns:
            results_df: 결과 데이터프레임
        """
        all_predictions = []
        all_filenames = []
        all_labels = []
        all_confidences = []
        all_logits = []

        # 기본은 classifier logits 평가.
        # 단, 체크포인트에 arcface_state_dict가 있으면 ArcFace logits 경로로 자동 전환한다.
        arcface_module = None
        model_dir = os.path.join(self.args.save_dir, self.args.project_name)
        checkpoint_path = getattr(self.args, '_current_eval_model_path', None)
        if checkpoint_path and os.path.exists(checkpoint_path):
            checkpoint = torch.load(checkpoint_path, map_location=self.args.device)
            arcface_state = checkpoint.get('arcface_state_dict') if isinstance(checkpoint, dict) else None
            arcface_cfg = checkpoint.get('arcface_config', {}) if isinstance(checkpoint, dict) else {}
            if arcface_state is not None and hasattr(model, 'embed_dim'):
                # 저장된 ArcFace 설정으로 모듈을 복원해 train-time 분류 경로와 일치시킨다.
                arcface_module = losses.SubCenterArcFace(
                    num_classes=int(arcface_cfg.get('num_classes', self.args.n_classes)),
                    feature_dim=int(arcface_cfg.get('feature_dim', model.embed_dim)),
                    num_sub_centers=int(arcface_cfg.get('num_sub_centers', getattr(self.args, 'num_sub_centers', 4))),
                    margin=float(arcface_cfg.get('margin', getattr(self.args, 'arcface_margin', 0.3))),
                    scale=float(arcface_cfg.get('scale', getattr(self.args, 'arcface_scale', 64.0))),
                ).to(self.args.device)
                arcface_module.load_state_dict(arcface_state, strict=True)
                arcface_module.eval()
                self.logger.info('[Eval] ArcFace checkpoint detected. Using ArcFace logits for prediction.')
        elif checkpoint_path is None and os.path.isdir(model_dir):
            self.logger.debug('[Eval] No _current_eval_model_path provided; fallback to classifier logits.')

        with torch.no_grad():
            for images, labels, paths in dataloader:
                images = images.to(self.args.device)

                outputs = model(images)
                if arcface_module is not None and hasattr(model, '_last_feature'):
                    # ArcFace checkpoint인 경우 feature->ArcFace logits를 사용한다.
                    features = F.normalize(model._last_feature, dim=1)
                    outputs = arcface_module.inference_logits(features)
                probs = torch.softmax(outputs, dim=1)
                _, preds = torch.max(outputs, 1)

                all_predictions.extend(preds.cpu().numpy().tolist())
                all_labels.extend(labels.numpy().tolist())
                all_filenames.extend(paths)
                all_confidences.extend(probs.cpu().numpy().tolist())
                all_logits.extend(outputs.cpu().numpy().tolist())

        self.logger.info(f"Total processed: {len(all_predictions)}")

        results_df = build_results_dataframe(
            file_paths=all_filenames,
            true_labels=all_labels,
            pred_labels=all_predictions,
            label_names=self.args.labels,
        )

        confidence_df = pd.DataFrame(
            all_confidences,
            columns=[f'confidence_{cls}' for cls in self.args.labels]
        )
        logits_df = pd.DataFrame(
            all_logits,
            columns=[f'logit_{cls}' for cls in self.args.labels]
        )

        class_score_df = pd.DataFrame(index=confidence_df.index)
        for cls in self.args.labels:
            class_score_df[f'confidence_{cls}'] = confidence_df[f'confidence_{cls}']
            class_score_df[f'logit_{cls}'] = logits_df[f'logit_{cls}']

        results_df = pd.concat([results_df, class_score_df], axis=1)

        idx_to_class = {idx: cls for idx, cls in enumerate(self.args.labels)}
        results_df['pred_confidence'] = results_df.apply(
            lambda row: row[f"confidence_{idx_to_class[row['pred_label']]}"],
            axis=1
        )
        results_df['pred_logit'] = results_df.apply(
            lambda row: row[f"logit_{idx_to_class[row['pred_label']]}"],
            axis=1
        )

        return results_df


# ============================================================================
# SMW 평가 엔진
# ============================================================================
class SMWEvalEngine(BaseEngine):
    """테스트 데이터셋에 대한 SMW 모델 평가 (56개 패치 분리 + 후처리)"""
    
    def run(self, model, filenames_total, label_total, save_overlay=True):
        """SMW 모델 평가
        
        Args:
            model: 학습된 모델
            filenames_total: 이미지 경로 리스트
            label_total: 이미지 레이블 리스트
            save_overlay: Overlay 이미지 저장 여부
        
        Returns:
            results_df: 결과 데이터프레임
        """
        all_predictions = []
        all_filenames = []
        all_labels = []
        all_scores = []

        with torch.no_grad():
            for file_path, label_name in tqdm(
                list(zip(filenames_total, label_total)),
                total=len(filenames_total),
                ascii=True,
                desc='SMW Eval'
            ):
                try:
                    # 패치 추출
                    _, patches, _ = SMW_extract_patches(
                        file_path,
                        patch_width=self.args.img_size[0],
                        stride=self.args.img_size[0],
                    )
                    patches = patches.to(self.args.device, dtype=torch.float32)

                    # 패치별 추론
                    outputs = model(patches)
                    patch_preds = torch.argmax(outputs, dim=1).cpu().numpy()
                    final_pred, final_score = SMW_get_result(patch_preds.copy())

                    # Overlay 이미지 저장
                    if save_overlay:
                        pred_label_name = self.args.labels[final_pred]
                        if pred_label_name == 'OK_OVERLAP':
                            pred_label_name = 'OK'

                        image_name = os.path.splitext(os.path.basename(file_path))[0]
                        # pred_images 폴더에 저장 (평가용)
                        pred_image_root = os.path.join(
                            self.args.save_dir,
                            self.args.project_name,
                            'eval_pred_images'
                        )
                        os.makedirs(os.path.join(pred_image_root, pred_label_name), exist_ok=True)
                        
                        save_path = os.path.join(
                            pred_image_root,
                            pred_label_name,
                            f"{image_name}_pred_{pred_label_name}.png"
                        )
                        SMW_get_pred_image(
                            pred=patch_preds,
                            img=file_path,
                            label=label_name,
                            save_path=save_path,
                            show=False,
                        )

                    all_predictions.append(final_pred)
                    all_labels.append(normalize_smw_label(self.args.labels.index(label_name)))
                    all_filenames.append(file_path)
                    all_scores.append(final_score)

                except Exception as e:
                    self.logger.error(f"Error processing {file_path}: {e}")
                    continue

        self.logger.info(f"Total processed: {len(all_predictions)}")

        return build_results_dataframe(
            file_paths=all_filenames,
            true_labels=all_labels,
            pred_labels=all_predictions,
            label_names=self.args.labels[:-1],
            extra_columns={'pred_score': all_scores},
        )


# ============================================================================
# 평가 매니저
# ============================================================================
class EvalManager:
    """평가 엔진 관리 및 조율"""
    
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
        
        # 평가 엔진 선택
        if is_smw:
            self.engine = SMWEvalEngine(args, self.logger)
        else:
            self.engine = StandardEvalEngine(args, self.logger)
    
    def run(self, filenames_total, label_total, dataloader=None, cm_display_order=None):
        """평가 실행
        
        Args:
            filenames_total: 이미지 경로 리스트
            label_total: 이미지 레이블 리스트
            dataloader: 데이터로더 (표준 모델용)
            cm_display_order: Confusion Matrix 표시 순서
        
        Returns:
            eval_results: 모델별 평가 결과 딕셔너리
        """
        self.logger.info(f"Dataset: {self.args.datasets_name}")
        self.logger.info(f"Total test samples: {len(filenames_total)}")

        # 모델 폴더의 모든 .pth 파일 순회
        model_dir = os.path.join(self.args.save_dir, self.args.project_name)
        model_files = sorted(glob.glob(os.path.join(model_dir, "*.pth")))

        if len(model_files) == 0:
            self.logger.error(f"No model files found in {model_dir}")
            return {}

        self.logger.info(f"Found {len(model_files)} model(s) to evaluate")

        # 모델별 평가
        eval_results = {}

        for model_path in model_files:
            model_name = os.path.splitext(os.path.basename(model_path))[0]

            self.logger.info(f"\n{'='*60}")
            self.logger.info(f"Evaluating model: {model_name}")
            self.logger.info(f"{'='*60}")

            try:
                # 모델 로드
                device = torch.device(self.args.device)
                # 엔진 내부에서 현재 체크포인트를 직접 읽어 ArcFace 유무를 판단할 수 있도록 경로 전달.
                self.args._current_eval_model_path = model_path
                model = load_model(self.args, model_path, device)
                self.logger.info(f"Model loaded successfully")

                # 평가 실행
                if self.is_smw:
                    results_df = self.engine.run(model, filenames_total, label_total)
                    summary_label_names = self.args.labels[:-1]
                    if cm_display_order is None:
                        cm_display_order = ['NG_NO', 'NG_OVER', 'NG_WEAK', 'NG_PINHOLE', 'NG_SPATTER', 'OK']
                    export_onnx_model(model, self.args, model_path, batch_size=56)
                else:
                    results_df = self.engine.run(model, dataloader)
                    summary_label_names = self.args.labels
                    if cm_display_order is None:
                        cm_display_order = [
                            'CRACK_FOIL', 'CRACK_1', 'CRACK_2', 'DEBRIS',
                            'PROTRUSION_1', 'PROTRUSION_2', 'CRATER', 'SCRATCH_TINY'
                        ]
                    export_onnx_model(model, self.args, model_path, batch_size=1)

                # 결과 통계 출력
                self.logger.info(f"\nEvaluation Results:")
                self.logger.info(f"\n{results_df['pred_label_name'].value_counts()}")

                # 성과 요약
                summary = summarize_classification_results(
                    results_df=results_df,
                    label_names=summary_label_names,
                    display_order=cm_display_order,
                )

                self.logger.info(f"Accuracy: {summary['accuracy']:.4f}")
                self.logger.info(f"F1 Score (Macro): {summary['f1_macro']:.4f}")
                self.logger.info(f"\n[ Confusion Matrix ]")
                self.logger.info(f"\n{summary['cm_df']}")

                # 결과 저장
                csv_path = os.path.join(
                    self.args.save_dir,
                    self.args.project_name,
                    os.path.basename(model_path) + '.csv'
                )
                save_results_dataframe(results_df, csv_path)

                eval_results[model_name] = {
                    'results_df': results_df,
                    'summary': summary,
                    'model_path': model_path,
                }

            except Exception as e:
                self.logger.error(f"Error evaluating model {model_name}: {e}")
                import traceback
                traceback.print_exc()
                continue

        self.logger.info(f"\n{'='*60}")
        self.logger.info(f"Evaluation completed! Evaluated {len(eval_results)} model(s)")
        self.logger.info(f"\n{'='*60}")

        return eval_results
