import os 
import numpy as np 
import random 

import torch
import cv2 
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
            return torch.clamp(noise, -0.5, 0.5)

        else: 
            return tensor


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
            transforms.ColorJitter(brightness=0.1, contrast=0.1),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.5, 0.5, 0.5], std=[1.0, 1.0, 1.0]),
        ])
    else:
        transform = transforms.Compose([
            transforms.Resize((size, size)),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.5, 0.5, 0.5], std=[1.0, 1.0, 1.0]),
        ])
    return transform