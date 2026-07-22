from tqdm import tqdm
import timm 
import os 
import glob
import numpy as np
import math
import pandas as pd 

import cv2
import torch
import torch.nn as nn

from torchvision import transforms
from PIL import Image

from train import custom_model


def extract_patches_overlap(img: Image.Image,
                            patch_size: int,
                            stride: int):
    """
    이미지에 right·bottom 패딩을 추가한 뒤
    stride만큼 이동하며 patch_size 크기의 패치 추출.
    반환: patches (PIL list), positions (x,y 리스트), n_cols, n_rows
    """
    w, h = img.size
    
    # 필요한 패딩 계산
    n_cols = math.ceil((w - patch_size) / stride) + 1
    n_rows = math.ceil((h - patch_size) / stride) + 1
    pad_w = max(0, (n_cols - 1) * stride + patch_size - w)
    pad_h = max(0, (n_rows - 1) * stride + patch_size - h)

    img_padded = transforms.Pad((0, 0, pad_w, pad_h))(img)
    patches, positions = [], []

    for row in range(n_rows):
        for col in range(n_cols):
            x = col * stride
            y = row * stride
            patch = img_padded.crop((x, y, x + patch_size, y + patch_size))
            patches.append(patch)
            positions.append((x, y))

    return patches, positions, n_cols, n_rows

def infer_with_overlap(CONFIG: dict,
                        image_path: str,
                       model,
                       CLAHE,
                       patch_size: int = 224,
                       stride: int = 112,
                       batch_size: int = 24, 
                       ):
    
    kernel = np.array([[0, -0.5, 0], [-0.5, 3,-0.5], [0, -0.5, 0]]) ## 적절하게 고주파 강조 

    _gray = Image.open(image_path).convert("L")
    _gray = cv2.filter2D(np.array(_gray), -1, kernel) 
    gray = CLAHE.apply(_gray)
    img = np.stack([gray] * 3, axis = -1)
    img = Image.fromarray(img, mode = "RGB")
    
    # ## 검사 사각지역 제거를 위해 하단 112 부근을 상단에 이어붙히도록 전처리 
    # img_crop = img.crop((0, img.height - 112, img.width, img.height))
    # new_img = Image.new("RGB", (img.width, img.height + 112))
    # new_img.paste(img_crop, (0, 0))
    # new_img.paste(img, (0, 112))

    patches, positions, n_cols, n_rows = extract_patches_overlap(
        img, patch_size, stride)
    

    # Tensor 변환 및 DataLoader 구축
    preprocess = transforms.Compose([transforms.ToTensor(), 
                                     transforms.Normalize(mean=[0.5, 0.5, 0.5], std=[1.0, 1.0, 1.0])])

    inputs = torch.stack([preprocess(p) for p in patches])
    
    loader = torch.utils.data.DataLoader(
        inputs, batch_size=batch_size, shuffle=False)

    # 추론
    all_probs = []
    all_logits = []
    with torch.no_grad():
        for batch in loader:
            batch = batch.to(CONFIG["device"])
            logits = model(batch)
            probs = torch.softmax(logits, dim=1)
            all_probs.append(probs.cpu())
            all_logits.append(logits.cpu())

    all_probs = torch.cat(all_probs, dim=0)  # [N, num_classes]
    
    # 패치별 확률을 (n_rows × n_cols × num_classes)로 정리
    probs_grid = all_probs.view(n_rows, n_cols, -1)
    
    return probs_grid, positions


def _get_path(SAVE): 

    OK_polar = os.path.join(SAVE, "OK", "polar")
    OK_csv = os.path.join(SAVE, "OK", "csv")
    NG_bounding_box = os.path.join(SAVE, "NG", "bounding_box")
    NG_csv = os.path.join(SAVE, "NG", "csv")

    for _p in [OK_polar, OK_csv, NG_bounding_box, NG_csv]: 
        os.makedirs(_p, exist_ok=True)


def _get_model(CONFIG, backbone_path, epoch_path, num_classes): 

    model = timm.create_model(CONFIG["model_name"], pretrained=False)
    weight = torch.load(backbone_path)
    model.load_state_dict(weight, strict=False)

    model.head = custom_model.TinyWithGeM(model.head.fc.in_features, num_classes=num_classes)

    model.load_state_dict(torch.load(epoch_path))

    print(CONFIG["model_name"])
    print(CONFIG["project_name"])
    
    return model 


def _get_data(ROOT): 

    TEST_LIST = glob.glob(os.path.join(ROOT, "*.bmp"))
    print(f"Test list number : {len(TEST_LIST)}")

    return TEST_LIST

def _get_judge_class(model, patches, threshold,  n_rows, n_cols, CONFIG) : 


    # Tensor 변환 및 DataLoader 구축
    preprocess = transforms.Compose([transforms.ToTensor(), 
                                        transforms.Normalize(mean=[0.5, 0.5, 0.5], std=[1.0, 1.0, 1.0])])
    inputs = torch.stack([preprocess(p) for p in patches])
    
    loader = torch.utils.data.DataLoader(
        inputs, batch_size=24, shuffle=False)

    # 추론
    all_probs = []
    all_logits = []

    with torch.no_grad():
        for batch in loader:
            batch = batch.to(CONFIG["device"])
            logits = model(batch)
            probs = torch.softmax(logits, dim=1)
            all_probs.append(probs.cpu())
            all_logits.append(logits.cpu())
    ## 결과 전처리 
    all_probs = torch.cat(all_probs, dim=0)  # [N, num_classes]

    probs_grid = all_probs.view(n_rows, n_cols, -1)
    probs_grid_avg = probs_grid.mean(axis= 1)

    if threshold != None : 

        cls123 = probs_grid_avg[:, 1:4]
        mask_123 = cls123 > threshold
        arg123 = torch.argmax(cls123, dim=1) + 1
        any_mask = mask_123.any(dim=1)
        class_grid = torch.where(any_mask, arg123, torch.zeros_like(arg123))
    else : 
        raise ValueError
    
    flatten_class = np.array(class_grid).flatten()

    return flatten_class, probs_grid_avg

def get_figure(img_cv, flatten_class, probs_grid_avg, positions) : 

    flatten_probs = np.array(probs_grid_avg[:,1]).flatten()
    _prob = np.sort(flatten_probs[flatten_class == 1])

    max_prob = _prob.max()
    
    if (flatten_class == 1).sum() > 1: ## 모든 box를 표기 
        max_prob =_prob[len(_prob) - (flatten_class == 1).sum()]

    true_grid = np.array([i for i in flatten_probs >= max_prob for _ in range(3)])
    folding_top_left = np.array(positions)[true_grid]

    for p in folding_top_left: 

        x, y = int(p[0]), int(p[1]) 
        top_left = (x,y)
        bottom_right = (x + 224, y + 224)
        cv2.rectangle(img_cv, top_left, bottom_right,(0, 255, 0), thickness=2)        

    return img_cv 

def _get_result(judge,  probs_grid_avg, judge_dict, f_name):

    ## 결과 csv
    probs = np.array(probs_grid_avg.reshape(8,4))

    result_df = pd.DataFrame({"OK_prob" : probs[:,0], "FOLDING_prob" : probs[:,1], "REFOR_prob" : probs[:,2], "TEARING_prob" : probs[:,3], "Patch_Class" : flatten_class})
    max_patch_idx = np.array(result_df['FOLDING_prob']).argmax()
    max_confidence = round(result_df['FOLDING_prob'][max_patch_idx] * 100, 2)

    judge_dict['CELL_ID'].append(f_name)
    judge_dict['JUDGE'].append(judge)
    judge_dict['PATCH'].append(max_patch_idx)
    judge_dict['CONFIDENCE'].append(max_confidence)


    return result_df, max_patch_idx, max_confidence, probs

def _save_result(): 

    return None 

def polar_inference(SAVE_path, epoch_path, backbone_path, data_path, CONFIG, num_classes, threshold): 

    TEST_LIST = _get_data(data_path)

    model = _get_model(CONFIG, backbone_path, epoch_path, num_classes)
    model.eval()

    ## 이미지 전처리 
    clahe = cv2.createCLAHE(clipLimit=2.5, tileGridSize=(14, 14))  ## Adaptive 평활화 
    kernel = np.array([[0, -0.5, 0], [-0.5, 3,-0.5], [0, -0.5, 0]]) ## 적절하게 고주파 강조 
    
    judge_dict = {'CELL_ID' : [], 
            'JUDGE' : [], 
            "PATCH" : [], 
            'CONFIDENCE' :[]}


    for img in tqdm.tqdm(TEST_LIST): 

        # img = r"C:\Users\administrator.DVC\Desktop\Jonghyeok\Work\NFF\FTF\Tab_Folding\DATA\_polar\polar_anode_0801_OK_MP1_valid\Polar_152756_1_OK.bmp"
        f_name = os.path.basename(img).split(".bmp")[0]
        _gray = Image.open(img).convert("L")
        _gray = cv2.filter2D(np.array(_gray), -1, kernel) 
        gray = clahe.apply(_gray)

        img = np.stack([gray] * 3, axis = -1)
        img = Image.fromarray(img, mode = "RGB")

        patches, positions, n_cols, n_rows = extract_patches_overlap(img, 224, 112)

        flatten_class, probs_grid_avg = _get_judge_class(model, patches, threshold,  n_rows, n_cols, CONFIG)

        Judge_list = []
        tmp_img = None
        if (flatten_class == 1).sum() > 0: 
            Judge_list.append(1)
            judge = 'NG'
            tmp_img = get_figure()

        elif (flatten_class == 3).sum() > 0 : 
            Judge_list.append(3)
            judge = 'TEARING'

        elif (flatten_class == 2).sum() > 0 : 
            Judge_list.append(2)
            judge = 'REFOR'

        else: 
            Judge_list.append(0)
            judge = 'OK'

        result_df, max_patch_idx, max_confidence, probs = _get_result(judge, probs_grid_avg, judge_dict, f_name)

        _save_result(judge, tmp_img, result_df, max_patch_idx, max_confidence, probs)