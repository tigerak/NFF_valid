import torch
import yaml, json

class argument:
    def __init__(self):
        self.model_name = 'm'
        self.batch_size = 32
        self.max_epoch = 60
        self.n_classes = 2
        self.num_workers = 0
        self.img_size =    (224, 224)
        self.scheduler = 'CosineAnnealingLR'
        self.lr           = 0.0005
        self.backbone_lr  = 0.000001 ## if finetunning
        self.min_lr       = 1e-6
        self.weight_decay = 0.001
        self.accumulation_steps = 2
        self.device =  "cuda:0" if torch.cuda.is_available() else "cpu"
        self.seed = 42
    
    def load(self, yaml_path):
        with open(yaml_path, 'r', encoding='utf-8') as f:
            config = yaml.load(f, Loader=yaml.FullLoader)
            for attr in config.keys():
                setattr(self, attr, config[attr])

    def save_json(self, json_path):
        with open(json_path, 'w', encoding='utf-8') as f:
            json.dump(self.__dict__, f, indent=4, ensure_ascii=False)
        print(f"Arguments saved to {json_path}")
    
    def to_dict(self):

        return self.__dict__.copy()