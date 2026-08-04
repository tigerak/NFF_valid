import os, time, glob
from pathlib import Path
from typing import List

# For data manipulation
import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split


def load_images_by_label(data_dir: Path, label: str, mode = 'train') -> List[str]:

    paths = []
    for ext in ('*.bmp', '*.jpg', '*.jpeg'):

        pattern = f"{label}/**/{ext}"
        paths.extend(glob.glob(os.path.join(data_dir, pattern), recursive=True))
    paths = sorted(set(paths))

    if (len(paths) == 0) & (mode == 'train') :

        raise ValueError(f"[Error] Training image len : 0 ")
        return -1
    else : 
        pass 

    return paths


def get_trainList(args):

    labels = args.labels
    train_dir = args.train_dir
    valid_source = str(getattr(args, 'valid_source', 'split_train')).lower()
    valid_random_state = int(getattr(args, 'valid_random_state', 42))

    train_image_data = {label: load_images_by_label(train_dir, label) for label in labels}

    # 전체 train 데이터셋 구성
    filenames_total = []
    label_total = []
    for label, images in train_image_data.items():
        filenames_total.extend(images)
        label_total.extend([label] * len(images))

    print(f"Total train images: {len(filenames_total)}")
    print("\n=== Train Label Distribution ===")
    print(pd.Series(label_total).value_counts())

    # 옵션 1) train에서 valid_ratio만큼 분리
    split_alias = {'split_train', 'split', 'train_split', 'train'}
    test_alias = {'test_dir', 'test'}

    if valid_source in split_alias:
        valid_ratio = float(getattr(args, 'valid_ratio', 0.1))
        if not (0.0 < valid_ratio < 1.0):
            raise ValueError(f"valid_ratio must be between 0 and 1, got {valid_ratio}")

        # 클래스 샘플 수가 너무 적으면 stratify가 실패할 수 있어 fallback 처리
        try:
            train_data, valid_data, train_labels, valid_labels = train_test_split(
                filenames_total,
                label_total,
                test_size=valid_ratio,
                stratify=label_total,
                random_state=valid_random_state,
            )
        except ValueError as e:
            print(f"[WARN] Stratified split failed: {e}")
            print("[WARN] Falling back to non-stratified split.")
            train_data, valid_data, train_labels, valid_labels = train_test_split(
                filenames_total,
                label_total,
                test_size=valid_ratio,
                stratify=None,
                random_state=valid_random_state,
            )

        print("\n=== Validation Source: split_train ===")
        print("\nTrain:")
        print(pd.Series(train_labels).value_counts())
        print("\nValid:")
        print(pd.Series(valid_labels).value_counts())
        return train_data, valid_data, train_labels, valid_labels

    # 옵션 2) test_dir 전체를 valid로 사용
    if valid_source in test_alias:
        test_dir = getattr(args, 'test_dir', None)
        if not test_dir:
            raise ValueError("valid_source='test_dir' requires args.test_dir")

        valid_image_data = {
            label: load_images_by_label(test_dir, label, mode='test')
            for label in labels
        }

        valid_data = []
        valid_labels = []
        for label, images in valid_image_data.items():
            valid_data.extend(images)
            valid_labels.extend([label] * len(images))

        if len(valid_data) == 0:
            raise ValueError(f"No validation images found under test_dir: {test_dir}")

        train_data = filenames_total
        train_labels = label_total

        print("\n=== Validation Source: test_dir ===")
        print(f"Valid images from test_dir: {len(valid_data)}")
        print("\nValid:")
        print(pd.Series(valid_labels).value_counts())
        return train_data, valid_data, train_labels, valid_labels

    raise ValueError(
        f"Unsupported valid_source: {valid_source}. "
        "Use one of ['split_train', 'test_dir']."
    )
