
import os 
import numpy as np 
import matplotlib.patches as patches
import matplotlib.pyplot as plt


from PIL import Image
from torchvision import transforms
import torch
import cv2 


def SMW_extract_patches(img, patch_width = 224, stride = 224):    
    DEVICE ='cuda' if torch.cuda.is_available() else 'cpu'
    tmp_img = Image.open(img).convert('RGB')

    infer_transform = transforms.Compose([
        transforms.Resize((224, 224*56)),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.5, 0.5, 0.5], std=[1.0, 1.0, 1.0])
        ])
    
    origin_transform =  transforms.Compose([
        transforms.Resize((100, 224*56)),
        transforms.ToTensor(),
        ])
    
    img_tensor = infer_transform(tmp_img).unsqueeze(0).to(DEVICE)
    origin_tensor = origin_transform(tmp_img).to(DEVICE)
    # img_tensor = torch.tensor(np.array(tmp_img)).permute(2, 0, 1).unsqueeze(0)

    _, C,H, W = img_tensor.shape 
    n_patches = (W- patch_width) // stride + 1 

    origins = []
    patches = []
    positions = []

    for i in range(n_patches): 
        x_start = i * stride 
        x_end = x_start + patch_width
        patch = img_tensor[:, :, :, x_start : x_end]
        origin = origin_tensor[ :, :, x_start : x_end]

        origins.append(origin)
        patches.append(patch)
        positions.append(x_start)

    patches = torch.cat(patches, dim = 0)

    return origins, patches, positions



def SMW_get_result(pred, 
                   threshold_dict =  {
                    0 : 0.0, ## NO Threshold
                    1 :0.0,  ## OVER Threshold
                    2 :0.0, ## WEAK Threshold
                    3 :0.0, ## PINHOLE Threshold
                    4 :0.0, ## SPATTER Threshold
                    }
                    ): 
    '''
    pred : 각 patch 별 추론 결과(56개)
    threshold_dict : 56개의 결과에 대한 최종 판정 threshold
    ex) 
        NO : 한 개 patch 이상일 경우 최종 불량 판정 
        OVER : 56개 중 10% 이상 불량일 경우 최종 불량 판정 


    '''
    result_arr = [0,0,0,0,0] ## 각 라벨의 Percentage를 저장
    
    pred[pred == 4] = 3  ## 7번은 6번으로 편입 
    
    total = len(pred)
    _unique = np.array(np.unique(pred, return_counts = True))
    
    ## 각 class의 값들을 계산 
    for i in range(_unique.shape[1]):
        # locals()[f'ratio_{_unique[0, i]}'] = round(_unique[1, i]/total, 3)
        result_arr[_unique[0, i]] = round(_unique[1, i]/total, 3)
    
        
    _sort = np.argsort(result_arr)[::-1]

    final_label = 3

    if (result_arr[0] > 0) | (result_arr[1] > 0) : ## 미용접과 핀홀은 한 개만 있어도 무조건 잡는다

        final_label = [0,1][np.argmax(np.array([result_arr[0],result_arr[1]]))]

        return (final_label, result_arr[final_label])

    if _sort[0] != 3: ## 만약 0번 인덱스가 6(정상)이 아니라면 -> 가장 많은 0번 인덱스가 불량임 
        final_label = _sort[0]
        
    elif _sort[0] == 3:  # 만약 0번 인덱스가 6이라면 

        is_NG = result_arr[_sort[1]] > threshold_dict[_sort[1]] #  1번 인덱스의 Percentage를 threshold와 비교


        if is_NG : ## threshold 초과 시 해당 불량이 불량이 됨
            final_label = _sort[1]
            
    return (final_label, result_arr[final_label])

import matplotlib.patches as patches


def SMW_get_pred_image(pred, img, label, save_path=None, show=False): 
    
    LABEL = ["NO",  "PHL", "SPTT", "OK", "OK_OVL"]
    tmp_image = cv2.imread(img)
    tmp_image = cv2.cvtColor(tmp_image, cv2.COLOR_BGR2RGB)

    tmp_image = cv2.resize(tmp_image, (224*56, tmp_image.shape[0]))
    
    patch_size = 224
    num_patches = len(pred)
    height, width, _ = tmp_image.shape

    colors = plt.cm.get_cmap('tab10', 5)  # 클래스 7개용

    # 5. 시각화용 빈 이미지 준비 (원본 크기 동일)
    overlay_image = tmp_image.copy()

    # 6. 패치별 색상 덧입히기
    fig, axes = plt.subplots(2, 1, figsize=(40, 1))  # 2행 1열 subplot

    # (1) 원본 이미지 시각화
    axes[0].imshow(tmp_image)
    axes[0].set_title(f"Label : {label} // f_name : {os.path.basename(img)}")
    axes[0].axis('off')

    # (2) 예측 색상 오버레이 시각화
    axes[1].imshow(tmp_image)

    # 각 패치에 대해 박스 추가
    for i in range(num_patches):
        x_start = i * patch_size
        rect = patches.Rectangle(
            (x_start, 0), patch_size, height,
            linewidth=0,
            edgecolor=None,
            facecolor=colors(pred[i]),
            alpha=0.4
        )
        axes[1].add_patch(rect)
        # 레이블 숫자도 중간에 추가
        axes[1].text(x_start + patch_size // 2, 3, LABEL[pred[i]], color='black', fontsize=10, ha='center')

    # 축 설정
    axes[1].set_xlim([0, width])
    axes[1].set_ylim([height, 0])
    axes[1].axis('off')

    
    plt.tight_layout()

    if save_path is not None:
        os.makedirs(os.path.dirname(save_path), exist_ok=True)
        plt.savefig(save_path, format='png', dpi=200, bbox_inches='tight', pad_inches=0)

    if show:
        plt.show()

    plt.close(fig)
    