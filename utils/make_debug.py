
from torch.utils.data import Dataset, DataLoader
import matplotlib.pyplot as plt
import cv2 
import numpy as np 

def show_image_from_loader(loader, class_names, num_batches, max_images = 30, window_name="dataloader_debug"):

    ## patch 확인 용 
    it = iter(loader)
    
    for b in range(num_batches):
        try:
            images, labels = next(it)
        except StopIteration:
            print("더 이상 batch가 없습니다.")
            break

        batch_size = images.shape[0]

        if max_images is not None:
            batch_size = min(batch_size, max_images)

        print(f"\n===== Batch {b} =====")
        
        for i in range(batch_size):
            img = images[i]

            # (C, H, W) -> (H, W, C)
            img = img.permute(1, 2, 0).numpy()

            # float -> uint8
            if img.dtype != np.uint8:
                img = (img + 0.5) * 255.0
                img = np.clip(img, 0, 255).astype(np.uint8)

            # RGB -> BGR
            img = cv2.cvtColor(img, cv2.COLOR_RGB2BGR)
            img = cv2.resize(img, (500, 500), interpolation=cv2.INTER_LINEAR)


            class_idx = int(labels[i].item())
            class_name = (
                class_names[class_idx]
                if class_idx < len(class_names)
                else "UNKNOWN"
            )

            # 화면에 텍스트 오버레이
            text = f"label: {class_idx} ({class_name})"
            cv2.putText(
                img,
                text,
                (10, 30),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.9,
                (0, 255, 0),
                2,
                cv2.LINE_AA,
            )

            print(f"[{i}] {text}")

            cv2.imshow(window_name, img)
            key = cv2.waitKey(0)

            if key == 27:  # ESC
                break

    cv2.destroyAllWindows()