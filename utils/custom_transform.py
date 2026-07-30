import os 
import io
import numpy as np 
import random 

import torch
import cv2 
from PIL import Image
from torchvision import transforms
from torchvision.transforms import InterpolationMode


## 랜덤으로 가우시안 노이즈 추가 
class AddGaussianNoise: 
    
    def __init__(self, p, mean = 0., std = 0.01):
        
        self.mean = mean 
        self.std = std 
        self.p = p 

    def __call__(self, tensor): 

        if random.random() < self.p :     
            noise = tensor + torch.randn_like(tensor) * self.std + self.mean
            return torch.clamp(noise, 0.0, 1.0)

        else: 
            return tensor


class RandomDownscaleRestore:
    """스케일을 줄였다가 원래 크기로 복원해 크롭 없이 크기 변화 효과를 줍니다."""

    def __init__(self, p=0.5, scale=(0.85, 1.0)):
        self.p = p
        self.scale = scale

    def __call__(self, img):
        if random.random() >= self.p:
            return img

        width, height = img.size
        scale_factor = random.uniform(self.scale[0], self.scale[1])
        resized_width = max(8, int(width * scale_factor))
        resized_height = max(8, int(height * scale_factor))

        small = img.resize((resized_width, resized_height), resample=Image.BILINEAR)
        return small.resize((width, height), resample=Image.BILINEAR)


class RandomCLAHE:
    """그레이스케일 특성이 강한 이미지에 지역 대비를 약하게 올립니다."""

    def __init__(self, p=0.25, clip_limit=(1.5, 3.0), tile_grid_size=(8, 8)):
        self.p = p
        self.clip_limit = clip_limit
        self.tile_grid_size = tile_grid_size

    def __call__(self, img):
        if random.random() >= self.p:
            return img

        gray = np.array(img.convert("L"))
        clip_limit = random.uniform(self.clip_limit[0], self.clip_limit[1])
        clahe = cv2.createCLAHE(clipLimit=clip_limit, tileGridSize=self.tile_grid_size)
        enhanced = clahe.apply(gray)
        enhanced_rgb = np.stack([enhanced] * 3, axis=-1)
        return Image.fromarray(enhanced_rgb, mode="RGB")


class RandomJPEGCompression:
    """저장 품질 차이와 압축 아티팩트를 약하게 모사합니다."""

    def __init__(self, p=0.2, quality=(60, 95)):
        self.p = p
        self.quality = quality

    def __call__(self, img):
        if random.random() >= self.p:
            return img

        quality = random.randint(self.quality[0], self.quality[1])
        buffer = io.BytesIO()
        img.save(buffer, format="JPEG", quality=quality)
        buffer.seek(0)
        return Image.open(buffer).convert("RGB")


def smw_transform(size, mode ='train'):
    if mode == 'train': 
        transform = transforms.Compose([
            transforms.Resize((size, size)),
            transforms.RandomHorizontalFlip(p=0.3),
            # transforms.RandomApply(
            #     [transforms.ColorJitter(brightness=0.02, contrast=0.02, saturation=0.02, hue=0.02)],
            #     p =.3),
            transforms.ToTensor(),  # 이미지를 PyTorch 텐서로 변환 (HWC -> CHW, [0, 1] 범위로 정규화)
            AddGaussianNoise(p = 0.2, mean = 0., std = 0.01),
            transforms.Normalize(mean=[0.5, 0.5, 0.5], std=[1.0, 1.0, 1.0]), 
            ])        

    else: 
        transform = transforms.Compose([
            transforms.Resize((size, size)),
            transforms.ToTensor(),  # 이미지를 PyTorch 텐서로 변환 (HWC -> CHW, [0, 1] 범위로 정규화)
            transforms.Normalize(mean=[0.5, 0.5, 0.5], std=[1.0, 1.0, 1.0]), 
            # transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
        ])     

    return transform    
   


def lhs_transform(size, p, mode ='train'):

    if mode == 'train': 
        transform = transforms.Compose([
            transforms.Resize((size, size)),  # 이미지 크기를 224x224로 조정
            transforms.RandomHorizontalFlip(p=0.3),
            transforms.RandomVerticalFlip(p=0.3),
            transforms.RandomApply([
                                transforms.RandomRotation(degrees=(-45, 45), interpolation=InterpolationMode.BILINEAR, fill=(255,255,255)),
                                # transforms.ColorJitter(brightness=0.1, contrast=0.1, saturation=0.1, hue=0.1),
                                ], p = p),
            transforms.ToTensor(),  # 이미지를 PyTorch 텐서로 변환 (HWC -> CHW, [0, 1] 범위로 정규화)
            AddGaussianNoise(p = p, mean = 0., std = 0.01),
            transforms.Normalize(mean=[0.5, 0.5, 0.5], std=[1.0, 1.0, 1.0])
            # transforms.RandomErasing(p=0.2, scale = (0.005, 0.02))
        ])

    else: 
        transform = transforms.Compose([
            transforms.Resize((size, size)),
            transforms.ToTensor(),  # 이미지를 PyTorch 텐서로 변환 (HWC -> CHW, [0, 1] 범위로 정규화)
            transforms.Normalize(mean=[0.5, 0.5, 0.5], std=[1.0, 1.0, 1.0])
        ])     

    return transform       


def folding_transform(size = 224 ,  max_epoch = None, epoch = None, mode ='train'):

    if mode == 'train' :
        
        ini_p = 0.8
        final_p = 0.1       
        current_p = ini_p - (ini_p - final_p) * (epoch / max_epoch)
        if current_p < 0.1 : current_p = 0.1

        print(f"Train augmentation p : {current_p}")
        transform = transforms.Compose([    
        transforms.Resize((size, size)),
        transforms.RandomHorizontalFlip(p=0.3),
        transforms.RandomVerticalFlip(p=0.2),
        transforms.RandomApply([transforms.RandomRotation(degrees=(-30, 30), interpolation=InterpolationMode.BILINEAR, fill=(0,0,0)),
                                transforms.RandomResizedCrop(size=size, scale=(0.9, 1.1), ratio=(0.9, 1.1)),
                                # transforms.RandomAffine(degrees=0, translate = (0.05, 0.05), scale = (0.95, 1.05)), 
                                transforms.GaussianBlur(kernel_size=3, sigma=(0.1, 2.0)),
                                transforms.ColorJitter(brightness=(0.3), contrast= (0.3)),
                                ], p = current_p),
        transforms.ToTensor(),  # 이미지를 PyTorch 텐서로 변환 (HWC -> CHW, [0, 1] 범위로 정규화)
        transforms.Normalize(mean=[0.5, 0.5, 0.5], std=[1.0, 1.0, 1.0]), 
        AddGaussianNoise(p = 0.5, mean = 0., std = 0.05),
        transforms.RandomErasing(p = 0.1, scale = (0.005, 0.005), ratio = (0.3, 3.3)),
        ])

    else: 
        transform = transforms.Compose([
            transforms.Resize((size, size)),
            transforms.ToTensor(),  # 이미지를 PyTorch 텐서로 변환 (HWC -> CHW, [0, 1] 범위로 정규화)
            transforms.Normalize(mean=[0.5, 0.5, 0.5], std=[1.0, 1.0, 1.0])
        ])     
    return transform         



def normal_transform(size, mode ='train'):

    if mode == 'train': 
        transform = transforms.Compose([
            transforms.Resize((size, size)),  # 이미지 크기를 224x224로 조정
            transforms.RandomHorizontalFlip(p=0.3),
            transforms.RandomVerticalFlip(p=0.3),
            transforms.ToTensor(),  # 이미지를 PyTorch 텐서로 변환 (HWC -> CHW, [0, 1] 범위로 정규화)
            transforms.Normalize(mean=[0.5, 0.5, 0.5], std=[1.0, 1.0, 1.0])
            # transforms.RandomErasing(p=0.2, scale = (0.005, 0.02))
        ])    
    else: 
        transform = transforms.Compose([
            transforms.Resize((size, size)),
            transforms.ToTensor(),  # 이미지를 PyTorch 텐서로 변환 (HWC -> CHW, [0, 1] 범위로 정규화)
            transforms.Normalize(mean=[0.5, 0.5, 0.5], std=[1.0, 1.0, 1.0])
        ])     

    return transform   


def surface_transform(size, mode='train'):
    """SURFACE 데이터셋용 transform (8bit 1ch 이미지 -> 3ch stacking 후 적용)"""
    if mode == 'train':
        transform = transforms.Compose([
            transforms.Resize((size, size)),

            transforms.RandomHorizontalFlip(p=0.5),
            transforms.RandomVerticalFlip(p=0.5),

            # 크롭 없이 스케일 변화
            RandomDownscaleRestore(p=0.5, scale=(0.85, 1.0)),

            # 색 정보보다 명암/질감 차이를 더 중시
            transforms.RandomApply([
                transforms.ColorJitter(brightness=0.15, contrast=0.20),
            ], p=0.5),

            # 경계/선명도 변화
            transforms.RandomApply([
                transforms.RandomAdjustSharpness(sharpness_factor=1.5, p=1.0),
            ], p=0.25),

            # 전체 대비를 약하게 보정하거나 균등화
            transforms.RandomChoice([
                transforms.RandomAutocontrast(p=1.0),
                transforms.RandomEqualize(p=1.0),
            ]),

            # 지역 대비 향상
            RandomCLAHE(p=0.25, clip_limit=(1.5, 3.0), tile_grid_size=(8, 8)),

            # 저장/전송 품질 편차
            RandomJPEGCompression(p=0.2, quality=(60, 95)),

            # 작은 결함을 지우지 않도록 블러는 약하게만
            transforms.RandomApply([
                transforms.GaussianBlur(kernel_size=3, sigma=(0.1, 0.8)),
            ], p=0.15),

            transforms.ToTensor(),
            AddGaussianNoise(p=0.15, mean=0., std=0.01),
            transforms.Normalize(mean=[0.5, 0.5, 0.5], std=[1.0, 1.0, 1.0]),
        ])
    else:
        transform = transforms.Compose([
            transforms.Resize((size, size)),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.5, 0.5, 0.5], std=[1.0, 1.0, 1.0]),
        ])
    return transform