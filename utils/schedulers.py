
import torch 
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F

from torch.optim import lr_scheduler


def get_optim_scheduler(args, model):
    
    optimizer = get_optimizer(args, model)

    if args.scheduler == 'CosineAnnealingLR':
        cosine_scheduler = lr_scheduler.CosineAnnealingLR(optimizer, T_max=args.max_epoch, 
                                                   eta_min=args.min_lr)
        cosine_scheduler = GradualWarmupScheduler(optimizer, multiplier= 1.2, total_epoch= 5, after_scheduler= cosine_scheduler)

    elif args.scheduler == None:
        return None, None
        
    return optimizer, cosine_scheduler


def get_optimizer(args, model) : 

    if args.finetuning == False : ## finetunning 미 활용

        trainable_params = [p for p in model.parameters() if p.requires_grad]
        print(f"[INFO] Trainable parameters: {sum(p.numel() for p in trainable_params):,} / Total: {sum(p.numel() for p in model.parameters()):,}")
        optimizer = optim.AdamW(trainable_params,
                                lr=args.lr, 
                                weight_decay=args.weight_decay)
        return optimizer
    

    elif (args.finetuning == True)& (args.datasets_name != 'FTF_FOLDING') : ## finetunning 활용 시(TinyVit)

        trainable_params = [p for p in model.parameters() if p.requires_grad]
        print(f"[INFO] Trainable parameters: {sum(p.numel() for p in trainable_params):,} / Total: {sum(p.numel() for p in model.parameters()):,}")
        optimizer = optim.AdamW(trainable_params,
                                lr=args.lr, 
                                weight_decay=args.weight_decay)
        
        return optimizer


    elif (args.finetuning == True) & (args.datasets_name  == 'FTF_FOLDING') : ## finetunning 활용 시(convNext)

        if args.model_name == "convnextv2_base.fcmae_ft_in22k_in1k": 
                
            # 파라미터를 그룹으로 분리
            # ConvNeXtV2의 GRN에서 gamma와 beta는 weight decay 제외
            no_decay = ['bias', 'gamma', 'beta', 'LayerNorm.weight', 'LayerNorm.bias', 'norm.weight', 'norm.bias']

            # Backbone과 Head를 분리
            backbone_params = []
            head_params = []
            no_decay_backbone = []
            no_decay_head = []
            
            grn_params_found = []
            
            for name, param in model.named_parameters():
                # GRN의 gamma와 beta는 weight decay 제외 (ConvNeXtV2에서 'gamma' 또는 'beta'가 이름에 포함)
                is_no_decay = any(nd in name for nd in no_decay)
                
                # Head인지 확인 (head, classifier, fc 등)
                is_head = any(h in name.lower() for h in ['head', 'classifier', 'fc'])
                
                if 'gamma' in name or 'beta' in name:
                    grn_params_found.append(name)
                
                if is_no_decay:
                    if is_head:
                        no_decay_head.append(param)
                    else:
                        no_decay_backbone.append(param)
                else:
                    if is_head:
                        head_params.append(param)
                    else:
                        backbone_params.append(param)
            
            # GRN 파라미터가 발견되었는지 확인 (선택적 출력)
            if grn_params_found:
                print(f"[INFO] Found {len(grn_params_found)} GRN parameters (gamma/beta) - weight decay excluded")
            
            # Fine-tuning을 위한 learning rate 설정
            backbone_lr = args.backbone_lr
            head_lr = args.lr

            # 파라미터 그룹 설정
            param_groups = []
            if backbone_params:
                param_groups.append({'params': backbone_params, 'lr': backbone_lr, 'weight_decay': args.weight_decay})
            if no_decay_backbone:
                param_groups.append({'params': no_decay_backbone, 'lr': backbone_lr, 'weight_decay': 0.0})
            if head_params:
                param_groups.append({'params': head_params, 'lr': head_lr, 'weight_decay': args.weight_decay})
            if no_decay_head:
                param_groups.append({'params': no_decay_head, 'lr': head_lr, 'weight_decay': 0.0})
            
            optimizer = optim.AdamW(param_groups)
            
            return optimizer
        
        elif args.model_name == "tf_efficientnetv2_m.in21k": 

            print("Not prepared")
            return -1
        
        else: 
            raise ValueError("Not define another model..(TinyVit)")


class GradualWarmupScheduler(torch.optim.lr_scheduler._LRScheduler):
   
   def __init__(self, optimizer, multiplier, total_epoch, after_scheduler=None):
       
       self.multiplier = multiplier
       self.total_epoch = total_epoch
       self.after_scheduler = after_scheduler
       self.finished = False
       super().__init__(optimizer)

   def get_lr(self):
       
       if self.last_epoch < self.total_epoch:
           # Warmup 단계
           return [base_lr * ((self.multiplier - 1.) * self.last_epoch / self.total_epoch + 1.)
                   for base_lr in self.base_lrs]

       else:
           
           # Warmup 끝나고 후속 scheduler로 넘김
           if self.after_scheduler:
               if not self.finished:
                   self.after_scheduler.base_lrs = [base_lr * self.multiplier for base_lr in self.base_lrs]
                   self.finished = True

               return self.after_scheduler.get_lr()
           
           return [base_lr * self.multiplier for base_lr in self.base_lrs]
       
   def step(self, epoch=None):
       
       if self.finished and self.after_scheduler:
           
           if epoch is None:
               self.after_scheduler.step(None)
           else:
               self.after_scheduler.step(epoch - self.total_epoch)
       else:
           super().step(epoch)

