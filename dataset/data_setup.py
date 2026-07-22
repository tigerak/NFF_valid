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

    LABELS = args.labels
    DATA_DIR = args.train_dir

    image_data = {label: load_images_by_label(DATA_DIR, label) for label in LABELS}
        
    # 전체 데이터셋 구성
    filenames_total = []
    label_total = []

    for label, images in image_data.items():
        filenames_total.extend(images)
        label_total.extend([label] * len(images))

    # 데이터 통계 출력
    print(f"Total: {len(filenames_total)}")

    # 라벨 분포 확인
    df_labels = pd.DataFrame({'label': label_total})

    print("\n=== Train Label Distribution ===")
    print(df_labels['label'].value_counts())

    # 층화 분할
    train_data, valid_data, train_labels, valid_labels = train_test_split(
        filenames_total, 
        label_total, 
        test_size = args.valid_ratio, 
        stratify = label_total, 
        random_state = 42
    )

    # 분할 후 라벨 분포
    print("\n=== After Split ===")
    print("\nTrain:")
    print(pd.Series(train_labels).value_counts())
    print("\nValid:")
    print(pd.Series(valid_labels).value_counts())

    return train_data, valid_data, train_labels, valid_labels
