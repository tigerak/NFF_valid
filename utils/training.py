import os, gc, time , copy 
from collections import defaultdict

import numpy as np
import pandas as pd

import torch
from torch.utils.data import Dataset, DataLoader, WeightedRandomSampler
from sklearn.metrics import f1_score, recall_score, accuracy_score

# Utils

from tqdm import tqdm
from colorama import Fore, Back, Style
b_ = Fore.BLUE
sr_ = Style.RESET_ALL


## custom
from utils import losses
from utils import make_debug
from utils import augmentation


def run_training(
                model,
                optimizer,
                scheduler,
                train_dataset,
                valid_dataset,
                args,
                logger=None
                ):
    if logger is None:
        logger = logging.getLogger(__name__)
        
    if torch.cuda.is_available():
        print("[INFO] Using GPU: {}\n".format(torch.cuda.get_device_name()))
    DEVICE = torch.device(args.device)

    # WeightedRandomSampler 적용 
    # 문자열 리스트 -> 정수 인덱스 변환 (문자열 아니고 정수이면 KeyError)
    labels_raw = getattr(train_dataset, "label", None)
    if labels_raw is None:
        labels_raw = getattr(train_dataset, "labels", None)
    if labels_raw is None:
        raise ValueError("train_dataset must have 'label' or 'labels'")

    train_label_indices = torch.tensor(
        [train_dataset.class_to_idx[lbl] for lbl in labels_raw],
        dtype=torch.long
    )

    # ========== WeightedRandomSampler 설정 ==========
    if args.use_weighted_sampler:
        # 클래스 별 샘플 수 계산
        num_classes = args.n_classes
        class_counts = torch.bincount(train_label_indices, minlength=num_classes).float()

        # 클래스 별 가중치 계산 (역수)
        class_weights = 1. / class_counts.clamp(min=1.)  # 최소값을 1로 설정하여 0으로 나누는 것을 방지
        class_weights = class_weights / class_weights.mean()

        # 샘플 별 가중치 할당
        sample_weights = class_weights[train_label_indices].double()

        sampler = WeightedRandomSampler(weights=sample_weights, num_samples=len(sample_weights), replacement=True)

        train_loader = DataLoader(
            train_dataset,
            batch_size=args.batch_size,
            num_workers=args.num_workers,
            sampler=sampler,
            drop_last=True,
            pin_memory=True,
            persistent_workers=(args.num_workers > 0)
        )
    else:
        train_loader = DataLoader(
            train_dataset,
            batch_size=args.batch_size,
            num_workers=args.num_workers,
            shuffle=True,
            drop_last=True,
            pin_memory=True,
            persistent_workers=(args.num_workers > 0)
        )
    # ========== WeightedRandomSampler 설정 끝 ==========

    valid_loader = DataLoader(
        valid_dataset,
        batch_size=args.batch_size,
        num_workers=args.num_workers,
        shuffle=False,
        drop_last=False,
        pin_memory=True,
        persistent_workers=(args.num_workers > 0)
    )
    # make_debug.show_image_from_loader(valid_loader, class_names=args.labels, num_batches = 20, max_images = 64)

    num_epochs = args.max_epoch
    
    start = time.time()
    best_model_wts = copy.deepcopy(model.state_dict())
    best_epoch_loss = np.inf
    history = defaultdict(list)
    
    
    save_root = os.path.join(args.save_dir, args.project_name)
    os.makedirs(save_root, exist_ok= True)
    
    args.save_json(os.path.join(os.path.join(save_root, 'train_arguments.json')))

    for epoch in range(1, num_epochs + 1): 
        gc.collect()
        

        ## if TabFolding 
        # train_dataset.set_transform(num_epochs, epoch)

        ## if SMW (p scheduling)
        if hasattr(train_dataset, 'p') and hasattr(train_dataset, 'patches'):
            current_p = 0.3 - (0.3 - 0.05) * (epoch / 50)
            if current_p < 0.05 : current_p = 0.05
            train_dataset.p = current_p
            print(f"Current Pinhole gen Prob : {current_p}...")

        accumulation_steps = getattr(args, 'accumulation_steps', 1)
        train_epoch_loss, train_recall, train_epoch_f1, train_epoch_acc = train_one_epoch(model, optimizer, scheduler,
                                           dataloader=train_loader, 
                                           device=DEVICE, epoch=epoch,
                                           args=args,
                                           accumulation_steps=accumulation_steps)

        val_epoch_loss, val_recall, val_epoch_f1, val_epoch_acc, _ = valid_one_epoch(model, optimizer, valid_loader,
                                                                                         device= DEVICE, epoch=epoch, num_classes =  args.n_classes)

        history['Train Loss'].append(train_epoch_loss)
        history['Valid Loss'].append(val_epoch_loss)
        history['Train Recall'].append(train_recall)
        history['Valid Recall'].append(val_recall)
        history['Train f1'].append(train_epoch_f1)
        history['Valid f1'].append(val_epoch_f1)
        history['Train_ACC'].append(train_epoch_acc)
        history['Valid ACC'].append(val_epoch_acc)



        # deep copy the model
        if best_epoch_loss > val_epoch_loss:

            print(f"{b_}Validation Loss decreased ({best_epoch_loss} ---> {val_epoch_loss})")
            best_epoch_loss = val_epoch_loss
            best_model_wts = copy.deepcopy(model.state_dict())

            PATH = "{}/f1_score{:.4f}_Loss{:.7f}_epoch{:.0f}.pth".format(save_root, val_epoch_f1, val_epoch_loss, epoch)
            torch.save(model.state_dict(), PATH)

            # Save a model file from the current directory
            print(f"Model Saved{sr_}")

        if epoch % 3 == 0:
        
            print(f"Model Saved{sr_}")
            PATH = "{}/f1_score{:.4f}_Loss{:.7f}_epoch{:.0f}.pth".format(save_root, val_epoch_f1, val_epoch_loss, epoch)

            torch.save(model.state_dict(), PATH)

        middle_time = time.time() - start
        print("Middle time: {:.0f}h {:.0f}m {:.0f}s".format(
            middle_time // 3600, (middle_time % 3600) // 60, (middle_time % 3600) % 60))
        print()

    end = time.time()
    time_elapsed = end - start
    print('Training complete in {:.0f}h {:.0f}m {:.0f}s'.format(
        time_elapsed // 3600, (time_elapsed % 3600) // 60, (time_elapsed % 3600) % 60))
    print("Best loss: {:.7f}".format(best_epoch_loss))

    # load best model weights
    model.load_state_dict(best_model_wts)

    if args.finetuning != False :
        model = model.merge_and_unload()  # LoRA 병합 및 메모리 해제

    PATH = "{}/Loss{:.7f}.pth".format(save_root, best_epoch_loss)
    torch.save(model.state_dict(), PATH)

    ## save 
    pd.DataFrame(copy.copy(history)).to_csv(os.path.join(save_root, 'train_history.csv'))
    return model, history



def train_one_epoch(model, optimizer, scheduler, dataloader, device, epoch, args, accumulation_steps=1):

    model.train()
    
    dataset_size = 0
    running_loss = 0.0
    running_recall  = 0.0
    running_f1 = 0.0 
    running_acc = 0.0

    bar = tqdm(enumerate(dataloader), total=len(dataloader), ascii= True)
    optimizer.zero_grad()  # 초기화

    # Mixup/Cutmix 설정
    use_mixup = getattr(args, 'use_mixup', False)
    use_cutmix = getattr(args, 'use_cutmix', False)
    mixup_alpha = getattr(args, 'mixup_alpha', 1.0)
    
    for step, data in bar:
        
        patch, label = data[0], data[1]
        images = patch.to(device, dtype=torch.float32)
        targets = label.to(device)
        
        batch_size = images.size(0)
        
        # ========== Mixup/Cutmix 적용 ==========
        if use_mixup and torch.rand(1).item() < 0.5:  # 50% 확률
            images, targets_a, targets_b, lam = augmentation.mixup(images, targets, alpha=mixup_alpha)
        elif use_cutmix and torch.rand(1).item() < 0.5:
            images, targets_a, targets_b, lam = augmentation.cutmix(images, targets, alpha=mixup_alpha)
        else:
            targets_a = targets
            targets_b = None
            lam = 1.0
        # =======================================

        outputs = model(images)
        
        focal_gamma = getattr(args, 'focal_gamma', 2.0)
        focal_alpha = getattr(args, 'focal_alpha', 0.25)

        # Mixup/Cutmix 손실 계산
        if targets_b is not None:
            loss_a = losses.focal_loss(outputs, targets_a, alpha=focal_alpha, gamma=focal_gamma)
            loss_b = losses.focal_loss(outputs, targets_b, alpha=focal_alpha, gamma=focal_gamma)
            loss = lam * loss_a + (1 - lam) * loss_b
        else:
            loss = losses.focal_loss(outputs, targets, alpha=focal_alpha, gamma=focal_gamma)
            # loss = custom_loss.loss_dynamic(outputs, targets, special_class=[1,4], special_weight=1.5, current_epoch=epoch, total_epochs=total_epoch)
            # loss = losses.criterion(outputs, targets)

        loss = loss / accumulation_steps  # Gradient Accumulation: loss 스케일링

        loss.backward()

        # Gradient Accumulation: accumulation_steps마다 optimizer step
        if (step + 1) % accumulation_steps == 0 or (step + 1) == len(dataloader):
            optimizer.step()
            optimizer.zero_grad()
            if scheduler is not None:
                scheduler.step()

        # 메트릭 계산 (targets_a 기준)
        targets_for_metric = targets_a if targets_b is not None else targets
        
        recall = recall_score(targets_for_metric.cpu().tolist(), torch.argmax(outputs, dim=1).cpu().tolist(), average='macro')
        f1 = f1_score(targets_for_metric.cpu().tolist(), torch.argmax(outputs, dim=1).cpu().tolist(), average='macro')
        acc = accuracy_score(targets_for_metric.cpu().tolist(), torch.argmax(outputs, dim=1).cpu().tolist())
        
        running_loss += (loss.item() * accumulation_steps * batch_size)  # 원래 loss 스케일 복원
        running_recall  += (recall * batch_size)
        running_f1  += (f1 * batch_size)
        running_acc  += (acc * batch_size)

        dataset_size += batch_size
        
        epoch_loss = running_loss / dataset_size
        epoch_recall = running_recall / dataset_size
        epoch_f1 = running_f1 / dataset_size
        epoch_acc = running_acc / dataset_size

        bar.set_postfix(Epoch=epoch, Train_Loss=epoch_loss, Train_recall = epoch_recall, Train_f1 = epoch_f1, train_acc = epoch_acc,
                        LR=optimizer.param_groups[0]['lr'])
    
    # gc.collect()
    
    return epoch_loss, epoch_recall, epoch_f1, epoch_acc

def valid_one_epoch(model, optimizer,dataloader, device, epoch, num_classes):
    with torch.inference_mode():

        model.eval()
        
        dataset_size = 0
        running_loss = 0.0
        running_recall  = 0.0
        running_f1 = 0.0 
        running_acc = 0.0

        class_losses = {i: [] for i in range(num_classes)}
        bar = tqdm(enumerate(dataloader), total=len(dataloader), ascii= True)
        for step, data in bar:        

            patch, label = data[0], data[1]
            images = patch.to(device, dtype=torch.float32)
            targets = label.to(device)
            
            batch_size = images.size(0)
            
            outputs = model(images)
            loss = losses.criterion(outputs, targets)

            losses.calculate_loss_per_class(class_losses, losses.criterion, outputs, targets)

            recall = recall_score(targets.cpu().tolist(), torch.argmax(outputs, dim=1).cpu().tolist(), average='macro')
            f1 = f1_score(targets.cpu().tolist(), torch.argmax(outputs, dim=1).cpu().tolist(), average='macro')
            acc = accuracy_score(targets.cpu().tolist(), torch.argmax(outputs, dim=1).cpu().tolist())

            
            running_loss += (loss.item() * batch_size)
            running_recall  += (recall * batch_size)
            running_f1  += (f1 * batch_size)
            running_acc  += (acc * batch_size)

            dataset_size += batch_size
            
            epoch_loss = running_loss / dataset_size
            epoch_recall = running_recall / dataset_size
            epoch_f1 = running_f1 / dataset_size
            epoch_acc = running_acc / dataset_size

            bar.set_postfix(Epoch=epoch, Valid_Loss=epoch_loss, Valid_recall=epoch_recall, Valid_f1 = epoch_f1, Valid_acc =  epoch_acc,
                            LR=optimizer.param_groups[0]['lr'])   
            

        avg_class_losses = {k: sum(v) / len(v) if len(v) > 0 else 'no samples' for k, v in class_losses.items()}
        print(f"Epoch {epoch+1}: {avg_class_losses}")
        gc.collect()
        
    return epoch_loss, epoch_recall, epoch_f1, epoch_acc, avg_class_losses



def train_one_epoch_contrastive(model, supcon_criterion, ce_criterion, loader, optimizer, device, alpha=0.5):
    model.train()
    total_loss = 0.0
    total_supcon = 0.0
    total_ce = 0.0
    total_correct = 0
    total_samples = 0

    progress_bar = tqdm(enumerate(loader), total=len(loader), desc="Training", leave=False)
    
    for batch_idx, (view1, view2, labels) in progress_bar:
        view1, view2, labels = view1.to(device), view2.to(device), labels.to(device)
        B = labels.size(0)

        # 두 뷰 합치기
        x = torch.cat([view1, view2], dim=0)  # [2B, C, H, W]
        feat, proj, logits = model(x)

        # contrastive input
        proj_a, proj_b = proj[:B], proj[B:]
        feats_for_supcon = torch.cat([proj_a, proj_b], dim=0)  # [2B, D]

        # losses
        loss_supcon = supcon_criterion(feats_for_supcon, labels)
        logits_for_ce = logits[:B]
        loss_ce = ce_criterion(logits_for_ce, labels)

        loss = alpha * loss_supcon + (1.0 - alpha) * loss_ce

        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        # 통계 업데이트
        total_loss += loss.item() * B
        total_supcon += loss_supcon.item() * B
        total_ce += loss_ce.item() * B
        total_correct += (logits_for_ce.argmax(dim=1) == labels).sum().item()
        total_samples += B

        avg_loss = total_loss / total_samples
        avg_supcon = total_supcon / total_samples
        avg_ce = total_ce / total_samples
        acc = total_correct / total_samples * 100

        # 진행 표시
        progress_bar.set_postfix(
            batch=f"{batch_idx+1}/{len(loader)}",
            loss=f"{avg_loss:.4f}",
            supcon=f"{avg_supcon:.4f}",
            ce=f"{avg_ce:.4f}",
            acc=f"{acc:.2f}%"
        )

    return total_loss / total_samples, total_supcon / total_samples, total_ce / total_samples, acc


def valid_one_epoch_contrastive(model, supcon_criterion, ce_criterion, loader, device, alpha=0.5):

    model.eval()
    total_loss = 0.0
    total_supcon = 0.0
    total_ce = 0.0
    total_correct = 0
    total_samples = 0

    progress_bar = tqdm(enumerate(loader), total=len(loader), desc="Validation", leave=False)
    
    with torch.no_grad():
        for batch_idx, (view1, view2, labels) in progress_bar:
            view1, view2, labels = view1.to(device), view2.to(device), labels.to(device)
            B = labels.size(0)

            # 두 뷰 합치기
            x = torch.cat([view1, view2], dim=0)  # [2B, C, H, W]
            feat, proj, logits = model(x)

            # contrastive input
            proj_a, proj_b = proj[:B], proj[B:]
            feats_for_supcon = torch.cat([proj_a, proj_b], dim=0)  # [2B, D]

            # losses
            loss_supcon = supcon_criterion(feats_for_supcon, labels)
            logits_for_ce = logits[:B]
            loss_ce = ce_criterion(logits_for_ce, labels)

            loss = alpha * loss_supcon + (1.0 - alpha) * loss_ce

            # 통계 업데이트
            total_loss += loss.item() * B
            total_supcon += loss_supcon.item() * B
            total_ce += loss_ce.item() * B
            total_correct += (logits_for_ce.argmax(dim=1) == labels).sum().item()
            total_samples += B

            avg_loss = total_loss / total_samples
            avg_supcon = total_supcon / total_samples
            avg_ce = total_ce / total_samples
            acc = total_correct / total_samples * 100

            # 진행 표시
            progress_bar.set_postfix(
                batch=f"{batch_idx+1}/{len(loader)}",
                loss=f"{avg_loss:.4f}",
                supcon=f"{avg_supcon:.4f}",
                ce=f"{avg_ce:.4f}",
                acc=f"{acc:.2f}%"
            )

    return total_loss / total_samples, total_supcon / total_samples, total_ce / total_samples, acc