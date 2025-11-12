
from pathlib import Path
from typing import Optional, Dict, Any
import json


class DatasetConfig:    
    def __init__(
        self,
        data_root: str = "/content/CV_Project/DAVIS-data/dataset",
        pretrain_dataset: str = "DUTS_train",
        pretrained_model_path: str = "/content/drive/MyDrive/CVProject_log/model/best_model.pth",
        train_main_dataset: str = "DAVIS_train",
        train_sub_dataset: Optional[str] = None,
        validation_dataset: str = "DAVIS_test"):
        self.data_root = data_root
        self.pretrain_dataset = pretrain_dataset
        self.pretrained_model_path = pretrained_model_path
        self.train_main_dataset = train_main_dataset
        self.train_sub_dataset = train_sub_dataset
        self.validation_dataset = validation_dataset
    
    def to_dict(self) -> Dict[str, Any]:
        return {
            'data_root': self.data_root,
            'pretrain': self.pretrain_dataset,
            'best_pretrained_model': self.pretrained_model_path,
            'DAVIS_train_main': self.train_main_dataset,
            'DAVIS_train_sub': self.train_sub_dataset,
            'DAVIS_val': self.validation_dataset}
    
    def validate_paths(self) -> bool:
        data_root_path = Path(self.data_root)
        
        if not data_root_path.exists():
            print(f"Warning: Data root does not exist: {self.data_root}")
            return False
        
        pretrain_path = data_root_path / self.pretrain_dataset
        if not pretrain_path.exists():
            print(f"Warning: Pretrain dataset does not exist: {pretrain_path}")
        
        main_train_path = data_root_path / self.train_main_dataset
        if not main_train_path.exists():
            print(f"Warning: Main training dataset does not exist: {main_train_path}")
        
        return True


class TrainingConfig:
    def __init__(
        self,
        gpu_devices: str = "0",
        num_epochs: int = 50,
        learning_rate: float = 1e-4,
        batch_size: int = 8,
        image_size: int = 384,
        num_slots: int = 2,
        warmup_epochs: int = 5,
        freeze_backbone_epochs: int = 2,
        patience: int = 7,
        val_every: int = 1,
        use_amp: bool = True
    ):
        self.gpu_devices = gpu_devices
        self.num_epochs = num_epochs
        self.learning_rate = learning_rate
        self.batch_size = batch_size
        self.image_size = image_size
        self.num_slots = num_slots
        self.warmup_epochs = warmup_epochs
        self.freeze_backbone_epochs = freeze_backbone_epochs
        self.patience = patience
        self.val_every = val_every
        self.use_amp = use_amp

    def to_dict(self) -> Dict[str, Any]:
        return {
            'GPU': self.gpu_devices,
            'epoch': self.num_epochs,
            'learning_rate': self.learning_rate,
            'batch_size': self.batch_size,
            'img_size': self.image_size,
            'slot_num': self.num_slots,
            'warmup_epochs': self.warmup_epochs,
            'freeze_backbone_epochs': self.freeze_backbone_epochs,
            'patience': self.patience,
            'val_every': self.val_every,
            'use_amp': self.use_amp,
        }

    def get_gpu_list(self):
        return [int(g.strip()) for g in self.gpu_devices.split(',')]


class ExperimentConfig:    
    def __init__(
        self,
        dataset_config: Optional[DatasetConfig] = None,
        training_config: Optional[TrainingConfig] = None):
        self.dataset = dataset_config or DatasetConfig()
        self.training = training_config or TrainingConfig()
    
    def to_dict(self) -> Dict[str, Dict[str, Any]]:
        return {
            'DATA': self.dataset.to_dict(),
            'TRAIN': self.training.to_dict()}
    
    def save_to_file(self, filepath: str):
        config_dict = self.to_dict()
        with open(filepath, 'w') as f:
            json.dump(config_dict, f, indent=4)
    
    @classmethod
    def load_from_file(cls, filepath: str):
        with open(filepath, 'r') as f:
            config_dict = json.load(f)
        
        dataset_cfg = DatasetConfig(**{
            'data_root': config_dict['DATA']['data_root'],
            'pretrain_dataset': config_dict['DATA']['pretrain'],
            'pretrained_model_path': config_dict['DATA']['best_pretrained_model'],
            'train_main_dataset': config_dict['DATA']['DAVIS_train_main'],
            'train_sub_dataset': config_dict['DATA']['DAVIS_train_sub'],
            'validation_dataset': config_dict['DATA']['DAVIS_val']})
        
        training_cfg = TrainingConfig(**{
            'gpu_devices': config_dict['TRAIN']['GPU'],
            'num_epochs': config_dict['TRAIN']['epoch'],
            'learning_rate': config_dict['TRAIN']['learning_rate'],
            'batch_size': config_dict['TRAIN']['batch_size'],
            'image_size': config_dict['TRAIN']['img_size'],
            'num_slots': config_dict['TRAIN']['slot_num']})
        
        return cls(dataset_cfg, training_cfg)
    
    def print_summary(self):
        print("=" * 60)
        print("EXPERIMENT CONFIGURATION")
        print("=" * 60)
        
        print("\nDataset Settings:")
        print(f"  Data Root: {self.dataset.data_root}")
        print(f"  Pretrain Dataset: {self.dataset.pretrain_dataset}")
        print(f"  Main Training: {self.dataset.train_main_dataset}")
        print(f"  Sub Training: {self.dataset.train_sub_dataset}")
        print(f"  Validation: {self.dataset.validation_dataset}")
        
        print("\nTraining Settings:")
        print(f"  GPU Devices: {self.training.gpu_devices}")
        print(f"  Epochs: {self.training.num_epochs}")
        print(f"  Learning Rate: {self.training.learning_rate}")
        print(f"  Batch Size: {self.training.batch_size}")
        print(f"  Image Size: {self.training.image_size}")
        print(f"  Number of Slots: {self.training.num_slots}")
        
        print("=" * 60)


DATA = {
    'data_root': "/content/CV_Project/DAVIS-data/dataset",
    'pretrain': "DUTS_train",
    'best_pretrained_model': "/content/drive/MyDrive/CVProject_log/model/best_model.pth",
    'DAVIS_train_main': "DAVIS_train",
    'DAVIS_train_sub': None, 
    'DAVIS_val': "DAVIS_test",
}

TRAIN = {
    'GPU': "0",
    'epoch': 50,
    'learning_rate': 1e-4,
    'batch_size': 8,
    'img_size': 384,
    'slot_num': 2,
    'warmup_epochs': 5,
    'freeze_backbone_epochs': 2,
    'patience': 7,
    'val_every': 1,
    'use_amp': True,
    'num_workers': 2
}
