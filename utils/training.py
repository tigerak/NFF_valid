import os, gc, time, copy 
import logging
from collections import defaultdict

import numpy as np
import pandas as pd

import torch
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader, WeightedRandomSampler

# Utils

from tqdm import tqdm

## custom
from utils import losses
from utils import make_debug
from utils import augmentation


def _update_confusion_matrix(conf_mat, targets, preds, num_classes):
    flat_idx = targets * num_classes + preds
    conf_mat += torch.bincount(flat_idx, minlength=num_classes * num_classes).reshape(num_classes, num_classes)


def _metrics_from_confusion_matrix(conf_mat):
    conf_mat = conf_mat.float()
    tp = torch.diag(conf_mat)
    support = conf_mat.sum(dim=1)
    predicted = conf_mat.sum(dim=0)

    recall_per_class = tp / support.clamp(min=1.0)
    precision_per_class = tp / predicted.clamp(min=1.0)
    f1_per_class = (2.0 * precision_per_class * recall_per_class) / (precision_per_class + recall_per_class).clamp(min=1e-12)

    macro_recall = recall_per_class.mean().item()
    macro_f1 = f1_per_class.mean().item()
    acc = (tp.sum() / conf_mat.sum().clamp(min=1.0)).item()

    return macro_recall, macro_f1, acc


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
        logger.info("[INFO] Using GPU: {}\n".format(torch.cuda.get_device_name()))
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

    # Stage1 loss 선택: focal | subcenter_arcface
    stage1_loss_mode = str(getattr(args, 'stage1_loss_mode', 'focal')).lower()
    use_sub_center_arcface = stage1_loss_mode == 'subcenter_arcface'
    sub_center_arcface = None
    if use_sub_center_arcface:
        num_sub_centers = int(getattr(args, 'num_sub_centers', 4))
        feature_dim = model.embed_dim if hasattr(model, 'embed_dim') else 768
        arcface_margin = float(getattr(args, 'arcface_margin', 0.3))
        arcface_scale = float(getattr(args, 'arcface_scale', 64.0))
        sub_center_arcface = losses.SubCenterArcFace(
            num_classes=args.n_classes,
            feature_dim=feature_dim,
            num_sub_centers=num_sub_centers,
            margin=arcface_margin,
            scale=arcface_scale
        )
        sub_center_arcface.to(DEVICE)
        logger.info(
            f'[Stage1] Sub-center ArcFace enabled: {num_sub_centers} sub-centers per class, '
            f'm={arcface_margin}, s={arcface_scale}'
        )
        print(f'[Stage1] Sub-center ArcFace: {num_sub_centers} centers, margin={arcface_margin}, scale={arcface_scale}')
    else:
        logger.info('[Stage1] Using Focal Loss')
        print('[Stage1] Using Focal Loss')

    # ArcFace 파라미터는 별도 optimizer로 업데이트
    if sub_center_arcface is not None:
        extra_optimizer = torch.optim.AdamW(
            sub_center_arcface.parameters(),
            lr=float(getattr(args, 'arcface_lr', 1e-3)),
            weight_decay=float(getattr(args, 'weight_decay', 0.0)),
        )
    else:
        extra_optimizer = None

    for epoch in range(1, num_epochs + 1): 
        gc.collect()
        

        ## if TabFolding 
        # train_dataset.set_transform(num_epochs, epoch)

        ## if SMW (p scheduling)
        if hasattr(train_dataset, 'p') and hasattr(train_dataset, 'patches'):
            current_p = 0.3 - (0.3 - 0.05) * (epoch / 50)
            if current_p < 0.05 : current_p = 0.05
            train_dataset.p = current_p
            logger.info(f"Current Pinhole gen Prob : {current_p}...")

        accumulation_steps = getattr(args, 'accumulation_steps', 1)
        train_epoch_loss, train_recall, train_epoch_f1, train_epoch_acc = train_one_epoch(
            model,
            optimizer,
            scheduler,
            dataloader=train_loader,
            device=DEVICE,
            epoch=epoch,
            args=args,
            accumulation_steps=accumulation_steps,
            sub_center_arcface=sub_center_arcface,
            extra_optimizer=extra_optimizer,
            logger=logger,
        )

        val_epoch_loss, val_recall, val_epoch_f1, val_epoch_acc, avg_class_losses = valid_one_epoch(
            model,
            optimizer,
            valid_loader,
            device=DEVICE,
            epoch=epoch,
            num_classes=args.n_classes,
            logger=logger,
        )

        # Sub-center ArcFace: 주기적 pruning (사용되지 않는 sub-center 제거)
        if sub_center_arcface is not None:
            prune_interval = int(getattr(args, 'arcface_prune_interval', 2))
            warmup_epochs = int(getattr(args, 'arcface_prune_warmup_epochs', 3))
            prune_min_usage = int(getattr(args, 'arcface_prune_min_usage', 2))
            prune_max_remove = int(getattr(args, 'arcface_prune_max_remove_per_class', 1))
            if epoch >= warmup_epochs and epoch % prune_interval == 0:
                removed = sub_center_arcface.prune_unused_subcenters(
                    warmup_epochs=warmup_epochs,
                    current_epoch=epoch,
                    min_usage=prune_min_usage,
                    max_remove_per_class=prune_max_remove,
                )
                if logger:
                    logger.info(f'[Epoch {epoch}] Sub-center pruning executed | removed={removed}')

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

            logger.info(f"Validation Loss decreased ({best_epoch_loss} -> {val_epoch_loss})")
            best_epoch_loss = val_epoch_loss
            best_model_wts = copy.deepcopy(model.state_dict())

            PATH = "{}/f1_score{:.4f}_Loss{:.7f}_epoch{:.0f}.pth".format(save_root, val_epoch_f1, val_epoch_loss, epoch)
            torch.save(model.state_dict(), PATH)

            # Save a model file from the current directory
            logger.info("Model Saved")

        if epoch % 3 == 0:
        
            logger.info("Model Saved")
            PATH = "{}/f1_score{:.4f}_Loss{:.7f}_epoch{:.0f}.pth".format(save_root, val_epoch_f1, val_epoch_loss, epoch)

            torch.save(model.state_dict(), PATH)

        logger.info(
            "Epoch %d/%d | train_loss=%.6f train_recall=%.4f train_f1=%.4f train_acc=%.4f | "
            "val_loss=%.6f val_recall=%.4f val_f1=%.4f val_acc=%.4f",
            epoch, num_epochs,
            train_epoch_loss, train_recall, train_epoch_f1, train_epoch_acc,
            val_epoch_loss, val_recall, val_epoch_f1, val_epoch_acc,
        )
        logger.info("Epoch %d class loss: %s", epoch, avg_class_losses)

        middle_time = time.time() - start
        logger.info("Middle time: {:.0f}h {:.0f}m {:.0f}s".format(
            middle_time // 3600, (middle_time % 3600) // 60, (middle_time % 3600) % 60))
        logger.info('')

    end = time.time()
    time_elapsed = end - start
    logger.info('Training complete in {:.0f}h {:.0f}m {:.0f}s'.format(
        time_elapsed // 3600, (time_elapsed % 3600) // 60, (time_elapsed % 3600) % 60))
    logger.info("Best loss: {:.7f}".format(best_epoch_loss))

    # load best model weights
    model.load_state_dict(best_model_wts)

    if args.finetuning != False :
        model = model.merge_and_unload()  # LoRA 병합 및 메모리 해제

    PATH = "{}/Loss{:.7f}.pth".format(save_root, best_epoch_loss)
    torch.save(model.state_dict(), PATH)

    ## save 
    pd.DataFrame(copy.copy(history)).to_csv(os.path.join(save_root, 'train_history.csv'))
    return model, history



def train_one_epoch(model, optimizer, scheduler, dataloader, device, epoch, args, accumulation_steps=1, sub_center_arcface=None, extra_optimizer=None, logger=None):

    model.train()
    
    dataset_size = 0
    running_loss = 0.0
    num_classes = args.n_classes
    conf_mat = torch.zeros((num_classes, num_classes), dtype=torch.long, device=device)

    bar = tqdm(enumerate(dataloader), total=len(dataloader), ascii= True)
    optimizer.zero_grad()  # 초기화

    # Mixup/Cutmix 설정
    use_mixup = getattr(args, 'use_mixup', False)
    use_cutmix = getattr(args, 'use_cutmix', False)
    if sub_center_arcface is not None:
        use_mixup = False
        use_cutmix = False
    mixup_alpha = getattr(args, 'mixup_alpha', 1.0)
    
    log_interval = int(getattr(args, 'log_interval', 100))
    arcface_fallback_notified = False

    for step, data in bar:
        
        patch, label = data[0], data[1]
        images = patch.to(device, dtype=torch.float32, non_blocking=True)
        targets = label.to(device, non_blocking=True)
        
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

        # Loss 계산: Sub-center ArcFace 또는 Focal Loss
        if sub_center_arcface is not None:
            # Sub-center ArcFace 사용 (feature 사용)
            if hasattr(model, '_last_feature'):
                try:
                    features = F.normalize(model._last_feature, dim=1)
                    loss_a = sub_center_arcface(features, targets_a)
                    if targets_b is not None:
                        loss_b = sub_center_arcface(features, targets_b)
                        loss = lam * loss_a + (1 - lam) * loss_b
                    else:
                        loss = loss_a
                except Exception as e:
                    # ArcFace 계산 중 예외가 나면 해당 step은 Focal Loss로 안전하게 진행
                    if not arcface_fallback_notified:
                        warn_msg = (
                            f"[경고] Epoch {epoch}: Sub-center ArcFace 계산 중 문제 발생으로 "
                            f"Focal Loss로 대체합니다. 원인: {e}"
                        )
                        print(warn_msg)
                        if logger is not None:
                            logger.warning(warn_msg)
                        arcface_fallback_notified = True

                    if targets_b is not None:
                        loss_a = losses.focal_loss(outputs, targets_a, alpha=focal_alpha, gamma=focal_gamma)
                        loss_b = losses.focal_loss(outputs, targets_b, alpha=focal_alpha, gamma=focal_gamma)
                        loss = lam * loss_a + (1 - lam) * loss_b
                    else:
                        loss = losses.focal_loss(outputs, targets, alpha=focal_alpha, gamma=focal_gamma)
            else:
                # feature 추출 실패 시 Focal Loss로 fallback
                if not arcface_fallback_notified:
                    warn_msg = (
                        f"[경고] Epoch {epoch}: Sub-center ArcFace용 feature(_last_feature) 추출 실패로 "
                        "Focal Loss로 대체합니다."
                    )
                    print(warn_msg)
                    if logger is not None:
                        logger.warning(warn_msg)
                    arcface_fallback_notified = True

                if targets_b is not None:
                    loss_a = losses.focal_loss(outputs, targets_a, alpha=focal_alpha, gamma=focal_gamma)
                    loss_b = losses.focal_loss(outputs, targets_b, alpha=focal_alpha, gamma=focal_gamma)
                    loss = lam * loss_a + (1 - lam) * loss_b
                else:
                    loss = losses.focal_loss(outputs, targets, alpha=focal_alpha, gamma=focal_gamma)
        else:
            # Focal Loss 사용 (기존 방식)
            if targets_b is not None:
                loss_a = losses.focal_loss(outputs, targets_a, alpha=focal_alpha, gamma=focal_gamma)
                loss_b = losses.focal_loss(outputs, targets_b, alpha=focal_alpha, gamma=focal_gamma)
                loss = lam * loss_a + (1 - lam) * loss_b
            else:
                loss = losses.focal_loss(outputs, targets, alpha=focal_alpha, gamma=focal_gamma)
        # ========== Mixup/Cutmix 적용 끝 ==========

        loss = loss / accumulation_steps  # Gradient Accumulation: loss 스케일링

        loss.backward()

        # Gradient Accumulation: accumulation_steps마다 optimizer step
        if (step + 1) % accumulation_steps == 0 or (step + 1) == len(dataloader):
            optimizer.step()
            if extra_optimizer is not None:
                extra_optimizer.step()
            optimizer.zero_grad()
            if extra_optimizer is not None:
                extra_optimizer.zero_grad()
            if scheduler is not None:
                scheduler.step()

        # 메트릭 계산 (targets_a 기준)
        targets_for_metric = targets_a if targets_b is not None else targets
        preds = torch.argmax(outputs, dim=1)
        _update_confusion_matrix(conf_mat, targets_for_metric.long(), preds.long(), num_classes)
        
        running_loss += (loss.item() * accumulation_steps * batch_size)  # 원래 loss 스케일 복원

        dataset_size += batch_size
        
        epoch_loss = running_loss / dataset_size
        bar.set_postfix(Epoch=epoch, Train_Loss=epoch_loss, LR=optimizer.param_groups[0]['lr'])

        if logger is not None and (step + 1) % max(log_interval, 1) == 0:
            logger.info(
                "[Train] epoch=%d step=%d/%d loss=%.6f lr=%.8f",
                epoch,
                step + 1,
                len(dataloader),
                epoch_loss,
                optimizer.param_groups[0]['lr'],
            )
    
    # gc.collect()
    
    epoch_recall, epoch_f1, epoch_acc = _metrics_from_confusion_matrix(conf_mat)

    return epoch_loss, epoch_recall, epoch_f1, epoch_acc

def valid_one_epoch(model, optimizer, dataloader, device, epoch, num_classes, logger=None):
    with torch.inference_mode():

        model.eval()
        
        dataset_size = 0
        running_loss = 0.0
        conf_mat = torch.zeros((num_classes, num_classes), dtype=torch.long, device=device)
        class_loss_sum = torch.zeros(num_classes, dtype=torch.float32, device=device)
        class_counts = torch.zeros(num_classes, dtype=torch.float32, device=device)
        bar = tqdm(enumerate(dataloader), total=len(dataloader), ascii= True)
        for step, data in bar:

            patch, label = data[0], data[1]
            images = patch.to(device, dtype=torch.float32, non_blocking=True)
            targets = label.to(device, non_blocking=True)
            
            batch_size = images.size(0)
            
            outputs = model(images)
            per_sample_loss = torch.nn.functional.cross_entropy(outputs, targets, reduction='none')
            loss = per_sample_loss.mean()
            preds = torch.argmax(outputs, dim=1)

            _update_confusion_matrix(conf_mat, targets.long(), preds.long(), num_classes)

            class_loss_sum.index_add_(0, targets.long(), per_sample_loss)
            class_counts.index_add_(0, targets.long(), torch.ones_like(per_sample_loss))

            
            running_loss += per_sample_loss.sum().item()

            dataset_size += batch_size
            
            epoch_loss = running_loss / dataset_size

            bar.set_postfix(Epoch=epoch, Valid_Loss=epoch_loss, LR=optimizer.param_groups[0]['lr'])

            if logger is not None:
                # valid는 배치 수가 상대적으로 적어서 마지막 step만 로그
                if (step + 1) == len(dataloader):
                    logger.info(
                        "[Valid] epoch=%d step=%d/%d loss=%.6f lr=%.8f",
                        epoch,
                        step + 1,
                        len(dataloader),
                        epoch_loss,
                        optimizer.param_groups[0]['lr'],
                    )
            
        epoch_recall, epoch_f1, epoch_acc = _metrics_from_confusion_matrix(conf_mat)

        avg_class_losses = {
            k: (class_loss_sum[k] / class_counts[k]).item() if class_counts[k] > 0 else 'no samples'
            for k in range(num_classes)
        }
        if logger is not None:
            logger.info(f"Epoch {epoch}: {avg_class_losses}")
        else:
            print(f"Epoch {epoch}: {avg_class_losses}")
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


# =============================================================================
# Stage-2: fine-tuned backbone + LSCE + optional KD
# =============================================================================

def _find_best_stage1_ckpt(save_root):
    """save_root 에서 f1_score*.pth 중 F1 최고 파일 경로 반환."""
    import glob as _glob
    candidates = _glob.glob(os.path.join(save_root, 'f1_score*_Loss*_epoch*.pth'))
    if not candidates:
        return None

    def _f1(p):
        try:
            return float(os.path.basename(p).split('f1_score')[1].split('_Loss')[0])
        except Exception:
            return -1.0

    return max(candidates, key=_f1)


def _build_stage2_optimizer(model, args):
    """백본 / head 에 각각 다른 LR 을 적용하는 옵티마이저 생성."""
    backbone_lr = float(getattr(args, 'stage2_backbone_lr', 1e-6))
    head_lr     = float(args.lr)
    wd          = float(args.weight_decay)

    backbone_params = [p for n, p in model.named_parameters()
                       if 'backbone' in n and p.requires_grad]
    head_params     = [p for n, p in model.named_parameters()
                       if 'backbone' not in n and p.requires_grad]

    param_groups = []
    if backbone_params:
        param_groups.append({'params': backbone_params, 'lr': backbone_lr, 'weight_decay': wd})
    if head_params:
        param_groups.append({'params': head_params, 'lr': head_lr, 'weight_decay': wd})

    return torch.optim.AdamW(param_groups)


def _set_stage2_trainable_blocks(model, args, logger=None):
    """Stage-2에서 backbone 마지막 블록만 학습 가능하도록 파라미터 제어.

    - stage2_train_scope: 'all_backbone' | 'last_blocks'
    - stage2_unfreeze_last_n_blocks: last_blocks일 때 해제할 블록 수
    """
    if not hasattr(model, 'backbone'):
        return

    scope = str(getattr(args, 'stage2_train_scope', 'last_blocks')).lower()
    unfreeze_n = int(getattr(args, 'stage2_unfreeze_last_n_blocks', 1))

    # 먼저 backbone 전체 freeze
    for p in model.backbone.parameters():
        p.requires_grad = False

    if scope == 'all_backbone':
        for p in model.backbone.parameters():
            p.requires_grad = True
        if logger is not None:
            logger.info('[Stage2] train_scope=all_backbone')
        return

    # 기본: last_blocks
    blocks = None
    if hasattr(model.backbone, 'blocks'):
        blocks = model.backbone.blocks
    elif hasattr(model.backbone, 'stages'):
        blocks = model.backbone.stages

    if blocks is None:
        # 백본 구조를 모르면 안전하게 전체 학습으로 fallback
        for p in model.backbone.parameters():
            p.requires_grad = True
        if logger is not None:
            logger.warning('[Stage2] backbone blocks not found, fallback to all_backbone')
        return

    n_total = len(blocks)
    n = max(1, min(unfreeze_n, n_total))
    for blk in list(blocks)[-n:]:
        for p in blk.parameters():
            p.requires_grad = True

    if logger is not None:
        logger.info('[Stage2] train_scope=last_blocks | unfreeze_last_n=%d / total_blocks=%d', n, n_total)


def run_stage2_training(model, optimizer, scheduler, train_dataset, valid_dataset, args, logger=None):
    """Stage-2: fine-tuned backbone + LSCE + optional KD (stage-1 teacher)."""
    import torch.nn.functional as _F
    from torch.optim import lr_scheduler as _lr_sched

    if logger is None:
        logger = logging.getLogger(__name__)

    device = torch.device(args.device)
    model.to(device)

    # Stage-2 학습 범위 설정: 기본은 마지막 블록만 학습
    _set_stage2_trainable_blocks(model, args, logger=logger)
    if hasattr(model, 'backbone'):
        model.backbone.train()

    if torch.cuda.is_available():
        logger.info('[Stage2] GPU: %s', torch.cuda.get_device_name())

    num_epochs     = int(getattr(args, 'stage2_max_epoch', 12))
    ls_eps         = float(getattr(args, 'stage2_label_smoothing', 0.05))
    kd_alpha       = float(getattr(args, 'stage2_kd_alpha', 0.5))
    kd_temperature = float(getattr(args, 'stage2_kd_temperature', 3.0))
    log_interval   = int(getattr(args, 'log_interval', 100))

    save_root = os.path.join(args.save_dir, args.project_name)
    os.makedirs(save_root, exist_ok=True)

    train_loader = DataLoader(
        train_dataset,
        batch_size=args.batch_size,
        num_workers=args.num_workers,
        shuffle=True, drop_last=True,
        pin_memory=True,
        persistent_workers=(args.num_workers > 0),
    )
    valid_loader = DataLoader(
        valid_dataset,
        batch_size=args.batch_size,
        num_workers=args.num_workers,
        shuffle=False, drop_last=False,
        pin_memory=True,
        persistent_workers=(args.num_workers > 0),
    )

    # ── 백본/head 분리 옵티마이저 + cosine scheduler ──────────────────────
    s2_optimizer = _build_stage2_optimizer(model, args)
    s2_scheduler = _lr_sched.CosineAnnealingLR(
        s2_optimizer, T_max=num_epochs, eta_min=float(args.min_lr)
    )

    # ── Teacher 로드 (KD용) ──────────────────────────────────────────────
    teacher = None
    teacher_weight = getattr(args, 'stage2_teacher_weight', '') or ''
    if not teacher_weight:
        teacher_weight = _find_best_stage1_ckpt(save_root) or ''

    if teacher_weight and os.path.exists(teacher_weight):
        teacher = copy.deepcopy(model)
        state = torch.load(teacher_weight, map_location=device)
        try:
            teacher.load_state_dict(state, strict=True)
        except RuntimeError as e:
            logger.error(
                '[Stage2] Teacher/Student 구조 불일치로 KD 학습을 중단합니다. '
                'token_fusion, head_type, n_classes, model_name 설정을 Stage1과 동일하게 맞추세요. '
                'teacher=%s | student_project=%s | reason=%s',
                teacher_weight,
                getattr(args, 'project_name', 'unknown'),
                str(e),
            )
            raise

        teacher.to(device).eval()
        for p in teacher.parameters():
            p.requires_grad = False
        logger.info('[Stage2] KD Teacher: %s', teacher_weight)
    else:
        kd_alpha = 0.0
        logger.warning('[Stage2] Teacher not found - KD disabled.')

    ce_criterion = torch.nn.CrossEntropyLoss(label_smoothing=ls_eps)

    backbone_lr = float(getattr(args, 'stage2_backbone_lr', 1e-6))
    train_scope = str(getattr(args, 'stage2_train_scope', 'last_blocks')).lower()
    unfreeze_n = int(getattr(args, 'stage2_unfreeze_last_n_blocks', 1))
    logger.info(
        '[Stage2] Start | epochs=%d  label_smoothing=%.3f  kd_alpha=%.2f  kd_T=%.1f  backbone_lr=%.2e  head_lr=%.2e  scope=%s  last_n=%d',
        num_epochs, ls_eps, kd_alpha, kd_temperature, backbone_lr, float(args.lr), train_scope, unfreeze_n,
    )

    best_f1        = -1.0
    best_model_wts = copy.deepcopy(model.state_dict())
    history        = defaultdict(list)
    start          = time.time()

    for epoch in range(1, num_epochs + 1):
        gc.collect()

        # ── Train ──────────────────────────────────────────────────────────
        model.train()
        dataset_size = 0
        running_loss = 0.0
        num_classes  = args.n_classes
        conf_mat     = torch.zeros((num_classes, num_classes), dtype=torch.long, device=device)
        bar          = tqdm(enumerate(train_loader), total=len(train_loader), ascii=True)
        s2_optimizer.zero_grad()

        for step, data in bar:
            images  = data[0].to(device, dtype=torch.float32, non_blocking=True)
            targets = data[1].to(device, non_blocking=True)
            batch_size = images.size(0)

            student_logits = model(images)
            ce_loss = ce_criterion(student_logits, targets)

            if teacher is not None and kd_alpha > 0.0:
                with torch.no_grad():
                    teacher_logits = teacher(images)
                T = kd_temperature
                kd_loss = _F.kl_div(
                    _F.log_softmax(student_logits / T, dim=1),
                    _F.softmax(teacher_logits / T, dim=1),
                    reduction='batchmean',
                ) * (T * T)
                total_loss = (1.0 - kd_alpha) * ce_loss + kd_alpha * kd_loss
            else:
                total_loss = ce_loss

            total_loss.backward()
            s2_optimizer.step()
            s2_optimizer.zero_grad()

            preds = torch.argmax(student_logits, dim=1)
            _update_confusion_matrix(conf_mat, targets.long(), preds.long(), num_classes)
            running_loss += total_loss.item() * batch_size
            dataset_size += batch_size
            epoch_loss    = running_loss / max(dataset_size, 1)

            bar.set_postfix(Epoch=epoch, S2_Loss=epoch_loss,
                            LR_bb=s2_optimizer.param_groups[0]['lr'])

            if logger is not None and (step + 1) % max(log_interval, 1) == 0:
                logger.info(
                    '[Stage2][Train] epoch=%d step=%d/%d loss=%.6f bb_lr=%.2e',
                    epoch, step + 1, len(train_loader), epoch_loss,
                    s2_optimizer.param_groups[0]['lr'],
                )

        s2_scheduler.step()
        train_recall, train_f1, train_acc = _metrics_from_confusion_matrix(conf_mat)

        # ── Validation ─────────────────────────────────────────────────────
        val_epoch_loss, val_recall, val_f1, val_acc, avg_class_losses = valid_one_epoch(
            model, s2_optimizer, valid_loader,
            device=device, epoch=epoch,
            num_classes=args.n_classes, logger=logger,
        )

        history['Train Loss'].append(epoch_loss)
        history['Valid Loss'].append(val_epoch_loss)
        history['Train F1'].append(train_f1)
        history['Valid F1'].append(val_f1)
        history['Train Recall'].append(train_recall)
        history['Valid Recall'].append(val_recall)
        history['Train ACC'].append(train_acc)
        history['Valid ACC'].append(val_acc)

        # ── Best 저장 (F1 기준) ────────────────────────────────────────────
        if val_f1 > best_f1:
            best_f1        = val_f1
            best_model_wts = copy.deepcopy(model.state_dict())
            best_path = os.path.join(
                save_root,
                f'stage2_f1_{best_f1:.4f}_Loss{val_epoch_loss:.7f}_epoch{epoch}.pth',
            )
            torch.save(model.state_dict(), best_path)
            logger.info('[Stage2] Best model saved: %s', best_path)

        logger.info(
            '[Stage2] Epoch %d/%d | '
            'train_loss=%.6f train_f1=%.4f train_acc=%.4f | '
            'val_loss=%.6f val_f1=%.4f val_acc=%.4f',
            epoch, num_epochs,
            epoch_loss, train_f1, train_acc,
            val_epoch_loss, val_f1, val_acc,
        )
        logger.info('[Stage2] Epoch %d class loss: %s', epoch, avg_class_losses)
        elapsed = time.time() - start
        logger.info('[Stage2] Middle time: %.0fh %.0fm %.0fs',
                    elapsed // 3600, (elapsed % 3600) // 60, elapsed % 60)
        logger.info('')

    elapsed = time.time() - start
    logger.info('[Stage2] Complete in %.0fh %.0fm %.0fs | best_val_F1=%.4f',
                elapsed // 3600, (elapsed % 3600) // 60, elapsed % 60, best_f1)

    model.load_state_dict(best_model_wts)
    final_path = os.path.join(save_root, f'stage2_final_f1_{best_f1:.4f}.pth')
    torch.save(model.state_dict(), final_path)
    logger.info('[Stage2] Final checkpoint: %s', final_path)

    pd.DataFrame(copy.copy(history)).to_csv(
        os.path.join(save_root, 'stage2_train_history.csv')
    )
    return model, history