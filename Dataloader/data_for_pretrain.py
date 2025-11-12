import os
from pathlib import Path
from PIL import Image, ImageEnhance, ImageOps
import torch.utils.data as data
import torchvision.transforms as transforms
import torchvision.transforms.functional as TF
import random
import numpy as np
import config as config


class DataAugmentation:    
    @staticmethod
    def flip_horizontal(img, mask, probability=0.5):
        if random.random() < probability:
            img = ImageOps.mirror(img)
            mask = ImageOps.mirror(mask)
        return img, mask
    
    @staticmethod
    def random_crop(img, mask, border=30):
        w, h = img.size
        crop_w = random.randint(w - border, w)
        crop_h = random.randint(h - border, h)
        left = (w - crop_w) // 2
        top = (h - crop_h) // 2
        right = left + crop_w
        bottom = top + crop_h
        return img.crop((left, top, right, bottom)), mask.crop((left, top, right, bottom))
    
    @staticmethod
    def rotate_random(img, mask, max_angle=15, probability=0.2):
        if random.random() > probability:
            angle = random.uniform(-max_angle, max_angle)
            img = img.rotate(angle, resample=Image.BICUBIC)
            mask = mask.rotate(angle, resample=Image.BICUBIC)
        return img, mask
    
    @staticmethod
    def adjust_colors(img):
        brightness = random.uniform(0.5, 1.5)
        contrast = random.uniform(0.5, 1.5)
        saturation = random.uniform(0.0, 2.0)
        sharpness = random.uniform(0.0, 3.0)
        img = ImageEnhance.Brightness(img).enhance(brightness)
        img = ImageEnhance.Contrast(img).enhance(contrast)
        img = ImageEnhance.Color(img).enhance(saturation)
        img = ImageEnhance.Sharpness(img).enhance(sharpness)
        return img
    
    @staticmethod
    def add_gaussian_noise(img, mean=0.1, std=0.35):
        arr = np.array(img, dtype=np.float32)
        noise = np.random.normal(mean, std, arr.shape)
        noisy = np.clip(arr + noise, 0, 255).astype(np.uint8)
        return Image.fromarray(noisy)
    
    @staticmethod
    def add_salt_pepper_noise(img, amount=0.0015):
        arr = np.array(img)
        num_pixels = int(amount * arr.shape[0] * arr.shape[1])
        
        for _ in range(num_pixels):
            x = random.randint(0, arr.shape[0] - 1)
            y = random.randint(0, arr.shape[1] - 1)
            arr[x, y] = 0 if random.random() < 0.5 else 255
        return Image.fromarray(arr)


class TrainingDataset(data.Dataset):
    
    def __init__(self, data_root=None, dataset_name=None, img_size=None):
        self.img_size = img_size or config.TRAIN['img_size']
        self.data_root = data_root or config.DATA['data_root']
        self.dataset_name = dataset_name or config.DATA['pretrain']
        base_path = Path(self.data_root) / self.dataset_name
        img_dir = base_path / "RGB"
        mask_dir = base_path / "GT"
        img_files = sorted(list(img_dir.glob("*.jpg")))
        mask_files = sorted(list(mask_dir.glob("*.jpg")) + list(mask_dir.glob("*.png")))
        self.samples = self._validate_pairs(img_files, mask_files)
        self.augmenter = DataAugmentation()
        self.img_transform = transforms.Compose([
            transforms.Resize((self.img_size, self.img_size)),
            transforms.ToTensor(),
            transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])])
        
        self.mask_transform = transforms.Compose([
            transforms.Resize((self.img_size, self.img_size)),
            transforms.ToTensor()])
    
    def _validate_pairs(self, img_paths, mask_paths):
        valid_pairs = []
        for img_path, mask_path in zip(img_paths, mask_paths):
            try:
                with Image.open(img_path) as img, Image.open(mask_path) as mask:
                    if img.size == mask.size:
                        valid_pairs.append((str(img_path), str(mask_path)))
            except Exception:
                continue
        return valid_pairs
    
    def _load_image(self, path, mode='RGB'):
        with open(path, 'rb') as f:
            return Image.open(f).convert(mode)
    
    def __getitem__(self, idx):
        img_path, mask_path = self.samples[idx]
        img = self._load_image(img_path, 'RGB')
        mask = self._load_image(mask_path, 'L')
        img, mask = self.augmenter.flip_horizontal(img, mask)
        img, mask = self.augmenter.random_crop(img, mask)
        img, mask = self.augmenter.rotate_random(img, mask)
        img = self.augmenter.adjust_colors(img)
        img = self.img_transform(img)
        mask = self.mask_transform(mask)
        
        return img, mask
    
    def __len__(self):
        return len(self.samples)


class ValidationDataset(data.Dataset):
    
    def __init__(self, data_root=None, dataset_name=None, img_size=None):
        self.img_size = img_size or config.TRAIN['img_size']
        self.data_root = data_root or config.DATA['data_root']
        self.dataset_name = dataset_name or config.DATA['DAVIS_val']

        self.samples = []
        base_path = Path(self.data_root) / self.dataset_name
        
        for subdir in sorted(base_path.iterdir()):
            if not subdir.is_dir():
                continue
            img_dir = subdir / "RGB"
            mask_dir = subdir / "GT"
            
            if not (img_dir.exists() and mask_dir.exists()):
                continue
            img_files = sorted(img_dir.glob("*.jpg"))
            mask_files = sorted(list(mask_dir.glob("*.jpg")) + list(mask_dir.glob("*.png")))
            
            for img_path, mask_path in zip(img_files, mask_files):
                self.samples.append((str(img_path), str(mask_path), subdir.name))

        self.img_transform = transforms.Compose([
            transforms.Resize((self.img_size, self.img_size)),
            transforms.ToTensor(),
            transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])])
        
        self.mask_transform = transforms.Compose([
            transforms.Resize((self.img_size, self.img_size)),
            transforms.ToTensor()])
    
    def _load_image(self, path, mode='RGB'):
        with open(path, 'rb') as f:
            return Image.open(f).convert(mode)
    
    def __getitem__(self, idx):
        img_path, mask_path, subdir_name = self.samples[idx]
        
        img = self._load_image(img_path, 'RGB')
        mask = self._load_image(mask_path, 'L')
        orig_size = mask.size
        filename = Path(img_path).stem + '.png'
        img_resized = img.resize((self.img_size, self.img_size))
        img_resized_arr = np.array(img_resized)
        img_tensor = self.img_transform(img)
        mask_tensor = self.mask_transform(mask)
        metadata = [orig_size, subdir_name, filename]
        
        return img_tensor, mask_tensor, metadata, img_resized_arr
    
    def __len__(self):
        return len(self.samples)


def create_train_loader(shuffle=True, num_workers=2, pin_memory=False):
    num_workers = config.TRAIN.get('num_workers', num_workers)
    dataset = TrainingDataset()
    return data.DataLoader(
        dataset=dataset,
        batch_size=config.TRAIN['batch_size'],
        shuffle=shuffle,
        num_workers=num_workers,
        pin_memory=pin_memory)


def create_test_loader(shuffle=False, num_workers=2, pin_memory=False): 
    num_workers = config.TRAIN.get('num_workers', num_workers)
    
    dataset = ValidationDataset()
    return data.DataLoader(
        dataset=dataset,
        batch_size=config.TRAIN['batch_size'],
        shuffle=shuffle,
        num_workers=num_workers,
        pin_memory=pin_memory)

get_loader = create_train_loader
get_testloader = create_test_loader