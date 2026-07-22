import os 
import numpy as np 
import random 
from pathlib import Path

from PIL import Image, ImageFilter
import cv2 

import torch
from torch.utils.data import Dataset, DataLoader
from torchvision import transforms
from torchvision.transforms import InterpolationMode

## custom
from utils import custom_transform


def get_dataset(train_data, valid_data, train_labels, valid_labels, args): 
    
    print(f"DataSet Name : {args.datasets_name}")
    print(f"Image Size : {args.img_size[0]}")


    if "ftf_folding" in args.datasets_name.lower():

        train_dataset = FoldingDataset(train_data, train_labels, args, mode = 'train')
        valid_dataset = FoldingDataset(valid_data, valid_labels, args, mode = 'valid')

        return train_dataset, valid_dataset
    
    elif "smw" in args.datasets_name.lower() : 

        train_dataset = SMWDataset(train_data, train_labels, args, mode = 'train')
        valid_dataset = SMWDataset(valid_data, valid_labels, args, mode = 'valid')

        return train_dataset, valid_dataset
    

    elif "lhs" in args.datasets_name.lower() : 
        train_dataset = LHSDataset(train_data, train_labels, args, mode = 'train')
        valid_dataset = LHSDataset(valid_data, valid_labels, args, mode = 'valid')

        return train_dataset, valid_dataset

    elif "surface" in args.datasets_name.lower() : 
        train_dataset = SURFACEDataset(train_data, train_labels, args, mode = 'train')
        valid_dataset = SURFACEDataset(valid_data, valid_labels, args, mode = 'valid')

        return train_dataset, valid_dataset
        
    
    elif any(keyword in args.datasets_name.lower() for keyword in ["ftf", "aia"]):
        train_dataset = NormalDataset(train_data, train_labels, args, mode = 'train')
        valid_dataset = NormalDataset(train_data, train_labels, args, mode = 'valid')

        return train_dataset, valid_dataset

    else:
        raise ValueError(f"Invalid dataset_name in argument(yaml) : {args.datasets_name}")
    


class NormalDataset(Dataset):
    def __init__(self,  filename_list, label_list, args, mode, transforms=None):
        
        self.classes = args.labels
        self.class_to_idx = {cls: idx for idx, cls in enumerate(self.classes)}  # 클래스 인덱스 매핑
        self.filename_list = filename_list
        self.label = label_list
        self.transforms = custom_transform.normal_transform(size = args.img_size[0],  mode = mode)

    def __len__(self):
        return len(self.label)
    
    def __getitem__(self, index):
        
        img_name = Path(self.filename_list[index])
        image = Image.open(img_name).convert("RGB")
        label = self.class_to_idx[self.label[index]]  # 언더스코어로 분리하여 첫 번째 부분을 라벨로 사용
        # 이미지가 PyTorch Tensor일 경우 NumPy 배열로 변환

        if self.transforms:
            image = self.transforms(image)        
        
        return image, torch.tensor(label).long(), self.filename_list[index]


class SURFACEDataset(Dataset):
    def __init__(self,  filename_list, label_list, args, mode, transforms=None):
        
        self.classes = args.labels
        self.class_to_idx = {cls: idx for idx, cls in enumerate(self.classes)}  # 클래스 인덱스 매핑
        self.filename_list = filename_list
        self.label = label_list
        self.transforms = custom_transform.surface_transform(size = args.img_size[0], mode = mode)

    def __len__(self):
        return len(self.label)
    
    def __getitem__(self, index):
        
        img_name = Path(self.filename_list[index])
        # 8bit 1channel grayscale 이미지 로드 후 3channel로 stacking
        gray = Image.open(img_name).convert("L")
        image = Image.merge("RGB", [gray, gray, gray])
        label = self.class_to_idx[self.label[index]]

        if self.transforms:
            image = self.transforms(image)        
        
        return image, torch.tensor(label).long(), self.filename_list[index]



class FoldingDataset(Dataset):
    def __init__(self, filename_list, label_list, args,  mode, transforms=None):
        
        self.classes = args.labels
        self.class_to_idx = {cls: idx for idx, cls in enumerate(self.classes)}  # 클래스 인덱스 매핑
        self.filename_list = filename_list
        self.label = label_list

        self.transforms = custom_transform.folding_transform(size = args.img_size[0], max_epoch = 100, epoch = 1, mode = mode)
        self.clahe = cv2.createCLAHE(clipLimit=2.5, tileGridSize=(14, 14))  ## Adaptive 평활화 
        self.kernel = np.array([[0, -0.5, 0], [-0.5, 3,-0.5], [0, -0.5, 0]]) ## 적절하게 고주파 강조 . 샤프닝

    def set_transform(self, max_epoch, epoch) : 
        self.transforms = custom_transform.folding_transform(224, max_epoch, epoch)

    def __len__(self):

        return len(self.label)

    def __getitem__(self, index):

        img_name = self.filename_list[index]

        _gray = Image.open(img_name).convert("L")
        _gray = cv2.filter2D(np.array(_gray), -1, self.kernel) 
        gray  = self.clahe.apply(_gray) 
        image = np.stack([gray] * 3, axis = -1)
        image = Image.fromarray(image, mode = "RGB")

        label = self.class_to_idx[self.label[index]]  
        image = self.transforms(image)             

        return image, torch.tensor(label).long()
    

class LHSDataset(Dataset):
    def __init__(self,  filename_list, label_list, args, mode, transforms=None):
        
        self.classes = args.labels
        self.class_to_idx = {cls: idx for idx, cls in enumerate(self.classes)}  # 클래스 인덱스 매핑
        self.filename_list = filename_list
        self.label = label_list
        self.p = 0.5

        self.kernel = ImageFilter.Kernel(size=(3, 3),
                                         kernel=[0, -1, 0, -1, 7, -1, 0, -1, 0], scale=None, offset=0)
        self.transforms = custom_transform.lhs_transform(size = args.img_size[0], p = self.p,  mode = mode)
        
    def __len__(self):
        return len(self.label)
    
    def __getitem__(self, index):
        
        img_name = self.filename_list[index]
        image = Image.open(img_name).convert("RGB")
        image = image.filter(self.kernel)

        label = self.class_to_idx[self.label[index]] 
        
        if self.transforms:
            image = self.transforms(image)        
                    
        return image, torch.tensor(label).long()
    

class SMWDataset(Dataset): 
    def __init__(self, filename_list, label_list, args, mode, patch_width = 224, stride = 112):
        self.filename_list = filename_list
        self.labels = label_list
        self.class_to_idx = {cls: idx for idx, cls in enumerate(args.labels)} 

        self.transforms = custom_transform.smw_transform(patch_width, mode = mode)

        self.patch_width = patch_width
        self.stride = stride
        self.patches = []

        self.p = 0.5
        
        for fname in self.filename_list:
            with Image.open(fname) as img:
                w, _ = img.size
                for x in range(0, w - self.patch_width + 1, self.stride):
                    self.patches.append((fname, x, self.class_to_idx[self.labels[self.filename_list.index(fname)]]))
        
    def __len__(self):
        return len(self.patches)
    
    def __getitem__(self, idx): 
        fname, x, label = self.patches[idx]

        with Image.open(fname) as img:
            
            if os.path.basename(fname).startswith("crop") :
                patch = img
                
            elif label == 1:

                label = 3 ## 원래는 정상인 라벨이었음 
                patch = img.crop((x, 0, x + self.patch_width, img.height))

                if random.random() < self.p:  

                    patch, center_y, axis_y = PHL_Augmentation(np.array(patch), num_holes=1, max_axes=10, max_angle=20)
                    label = pinhole_labeling(img.height, center_y, axis_y)

            else : 
                patch = img.crop((x, 0, x + self.patch_width, img.height))
                

            if self.transforms:
                patch = self.transforms(patch)

        return patch, label


### smw 전용 
def PHL_Augmentation(image, num_holes=1, max_axes=5, max_angle=180):

    output = image.copy()
    h, w, _ = image.shape
    mask = np.zeros((h, w), dtype=np.uint8)

    for _ in range(num_holes):
        
        center_x = random.randint(0, w - 10)

        if random.random() < 0.25:     ## 전 영역에 Random으로 핀홀을 형성하도록 
            center_y = random.randint(5, h-5)
            axis_x = random.randint(5, 15)
            axis_y = random.randint(3, max_axes)

        else: ## 중앙부에 핀홀을 형성하도록
            center_y = random.randint(int(h*0.25), int(h - h*0.50))
            axis_x = random.randint(3, 13)
            axis_y = random.randint(3, max_axes)
        
        angle = random.uniform(-max_angle, max_angle)
        
        ## step 1 타원 생성 
        temp_mask = np.zeros_like(mask)
        cv2.ellipse(temp_mask, (center_x, center_y), (axis_x, axis_y), angle, 0, 360, 255, -1)
        
        # Step 2: 마스크에 경계 노이즈 추가 (윤곽선 찾아서 변형)
        contours, _ = cv2.findContours(temp_mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        if contours:
            cnt = contours[0]
            irregular_cnt = []
            for pt in cnt:
                x, y = pt[0]
                x += random.randint(-2, 2)
                y += random.randint(-2, 2)
                irregular_cnt.append([[x, y]])
            irregular_cnt = np.array(irregular_cnt, dtype=np.int32)

            cv2.drawContours(mask, [irregular_cnt], -1, 255, -1)

    # Step 3: 가장자리 블러 처리 + 3채널 마스크 생성
    blurred_mask = cv2.GaussianBlur(mask, (5, 5), sigmaX=2)
    blurred_mask_3ch = cv2.merge([blurred_mask] * 3) / 255.0

    # Step 4: 노이즈 추가한 어둡게 처리
    noise = np.random.randint(0, 17, image.shape, dtype=np.uint8) ## 현재 Best 17.. 15, 13// 애매한 핀홀과 햇갈리는 경우 존재 
    darkened = (output * (1 - blurred_mask_3ch)).astype(np.uint8)
    noisy_dark = cv2.add(darkened, (noise * blurred_mask_3ch).astype(np.uint8))

    # Step 5: 원본 이미지와 섞기
    final = (output * (1 - blurred_mask_3ch) + noisy_dark * blurred_mask_3ch).astype(np.uint8)
    output = Image.fromarray(final)

    return output, center_y, axis_y


### smw 전용 
def pinhole_labeling(h, center_y, axis_y): 

    FINAL_CLASS = 1

    _top = center_y - axis_y
    _bottom = center_y + axis_y

    ## 높이 기준 20% 안쪽에 해당하는 영역이 진선핀홀 생성 영역임 
    pinhole_range = [h - h*0.35, h*0.2] 

    ## 만약 핀홀이 아래에 생성되거나 위에 생성되어 Spatter로 분류해야 하는 경우 
    if (_bottom > pinhole_range[0]) | (_top < pinhole_range[1]) :
        
        FINAL_CLASS = 2 ## spatter 

    return FINAL_CLASS



class SupConDataset(Dataset):
    def __init__(self, filename_list, label_list, LABELS,  mode, transforms=None):
        
        self.classes = LABELS
        self.class_to_idx = {cls: idx for idx, cls in enumerate(self.classes)}  # 클래스 인덱스 매핑
        self.filename_list = filename_list
        self.label = label_list
        self.transforms = custom_transform.folding_transform(size = 224, max_epoch = 100, epoch = 1, mode = mode)
        self.clahe = cv2.createCLAHE(clipLimit=2.5, tileGridSize=(14, 14))  ## Adaptive 평활화 
        self.kernel = np.array([[0, -0.5, 0], [-0.5, 3,-0.5], [0, -0.5, 0]]) ## 적절하게 고주파 강조 . 샤프닝

    def set_transform(self, max_epoch, epoch) : 
        self.transforms = custom_transform.folding_transform(224, max_epoch, epoch)

    def __len__(self):
        return len(self.label)
    
    def __getitem__(self, index):
        
        img_name = self.filename_list[index]

        _gray = Image.open(img_name).convert("L")
        _gray = cv2.filter2D(np.array(_gray), -1, self.kernel) 
        gray  = self.clahe.apply(_gray) 
        image = np.stack([gray] * 3, axis = -1)
        image = Image.fromarray(image, mode = "RGB")
        
        label = self.class_to_idx[self.label[index]]  
    
        if self.transforms:
            view1 = self.transforms(image)
            view2 = self.transforms(image)
        
        return view1, view2, torch.tensor(label).long()


