import os
from pathlib import Path
from PIL import Image, ImageEnhance, ImageOps
import torch
import torch.utils.data as data
import torchvision.transforms as transforms
import random
import numpy as np
import config as config
from typing import Dict, List, Tuple


class VideoAugmentation:
    @staticmethod
    def flip_horizontal(img, mask, flow, probability=0.5):
        if random.random() < probability:
            img = ImageOps.mirror(img)
            mask = ImageOps.mirror(mask)
            flow = ImageOps.mirror(flow)
        return img, mask, flow
    
    @staticmethod
    def random_crop(img, mask, flow, border=30):
        w, h = img.size
        crop_w = random.randint(w - border, w)
        crop_h = random.randint(h - border, h)
        
        left = (w - crop_w) // 2
        top = (h - crop_h) // 2
        right = left + crop_w
        bottom = top + crop_h
        
        bbox = (left, top, right, bottom)
        return img.crop(bbox), mask.crop(bbox), flow.crop(bbox)
    
    @staticmethod
    def rotate_random(img, mask, flow, max_angle=15, probability=0.2):
        if random.random() > probability:
            angle = random.uniform(-max_angle, max_angle)
            img = img.rotate(angle, resample=Image.BICUBIC)
            mask = mask.rotate(angle, resample=Image.BICUBIC)
            flow = flow.rotate(angle, resample=Image.BICUBIC)
        return img, mask, flow
    
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


class VideoSequenceDataset(data.Dataset):
    
    def __init__(self, main_dataset_name=None, sub_dataset_name=None, img_size=None):
        self.img_size = img_size or config.TRAIN['img_size']
        self.data_root = Path(config.DATA['data_root'])
        self.main_dataset = main_dataset_name or config.DATA['DAVIS_train_main']
        self.sub_dataset = sub_dataset_name or config.DATA['DAVIS_train_sub']
        self.augmenter = VideoAugmentation()
        
        main_base = self.data_root / self.main_dataset
        self.main_samples = self._load_dataset_files(main_base)
        self.main_class_index = self._build_class_index(
            [s[0] for s in self.main_samples],
            [s[2] for s in self.main_samples])
        
        self.sub_samples = []
        self.sub_class_index = {}
        
        if self.sub_dataset is not None:
            sub_base = self.data_root / self.sub_dataset
            self.sub_samples = self._load_dataset_files(sub_base)
            self.sub_class_index = self._build_class_index(
                [s[0] for s in self.sub_samples],
                [s[2] for s in self.sub_samples])
            
            self.sub_samples = self._balance_dataset(
                self.sub_samples, 
                len(self.main_samples))
        self.all_samples = self.main_samples + self.sub_samples
        
        self.img_transform = transforms.Compose([
            transforms.Resize((self.img_size, self.img_size)),
            transforms.ToTensor(),
            transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])])
        
        self.mask_transform = transforms.Compose([
            transforms.Resize((self.img_size, self.img_size)),
            transforms.ToTensor()])
        
        self.flow_transform = transforms.Compose([
            transforms.Resize((self.img_size, self.img_size)),
            transforms.ToTensor()])
    
    def _load_dataset_files(self, base_path: Path) -> List[Tuple[str, str, str]]:
        img_dir = base_path / 'RGB'
        mask_dir = base_path / 'GT'
        flow_dir = base_path / 'FLOW'
        img_files = sorted(img_dir.glob('*.jpg'))
        mask_files = sorted(list(mask_dir.glob('*.jpg')) + list(mask_dir.glob('*.png')))
        flow_files = sorted(list(flow_dir.glob('*.jpg')) + list(flow_dir.glob('*.png')))
        
        samples = [(str(img), str(mask), str(flow)) 
                   for img, mask, flow in zip(img_files, mask_files, flow_files)]
        
        return samples
    
    def _build_class_index(self, img_paths: List[str], flow_paths: List[str]) -> Dict[str, Dict[str, List[str]]]:
        class_index = {}
        
        for img_path, flow_path in zip(img_paths, flow_paths):
            class_name = self._extract_class_name(img_path)
            
            if class_name not in class_index:
                class_index[class_name] = {'images': [], 'flows': []}
            class_index[class_name]['images'].append(img_path)
            class_index[class_name]['flows'].append(flow_path)
        
        return class_index
    
    @staticmethod
    def _extract_class_name(filepath: str) -> str:
        filename = Path(filepath).stem
        return filename.split('_')[0]
    
    @staticmethod
    def _balance_dataset(samples: List, target_size: int) -> List:
        shuffled = samples.copy()
        random.shuffle(shuffled)
        return shuffled[:target_size]
    
    def _load_image(self, path: str, mode: str = 'RGB') -> Image.Image:
        with open(path, 'rb') as f:
            return Image.open(f).convert(mode)
    
    def _get_reference_frames(self, class_name: str, num_refs: int = 4) -> Tuple[List, List]:
        if class_name in self.main_class_index:
            class_data = self.main_class_index[class_name]
        else:
            class_data = self.sub_class_index[class_name]
        
        ref_imgs = class_data['images']
        ref_flows = class_data['flows']
        
        pairs = list(zip(ref_imgs, ref_flows))
        sampled_pairs = random.sample(pairs, min(num_refs, len(pairs)))
        
        ref_images, ref_flows = zip(*sampled_pairs)
        return list(ref_images), list(ref_flows)
    
    def _process_reference_frames(self, ref_img_paths: List[str], ref_flow_paths: List[str]) -> Tuple[torch.Tensor, torch.Tensor]:
        ref_img_tensors = []
        ref_flow_tensors = []
        
        for img_path, flow_path in zip(ref_img_paths, ref_flow_paths):
            img = self._load_image(img_path, 'RGB')
            flow = self._load_image(flow_path, 'RGB')
            
            img, _, flow = self.augmenter.flip_horizontal(img, img, flow)
            img, _, flow = self.augmenter.random_crop(img, img, flow)
            img, _, flow = self.augmenter.rotate_random(img, img, flow)
            img = self.augmenter.adjust_colors(img)
            
            img_tensor = self.img_transform(img)
            flow_tensor = self.flow_transform(flow)
            
            ref_img_tensors.append(img_tensor)
            ref_flow_tensors.append(flow_tensor)
        ref_imgs = torch.cat(ref_img_tensors, dim=0)
        ref_flows = torch.cat(ref_flow_tensors, dim=0)
        
        return ref_imgs, ref_flows
    
    def __getitem__(self, idx: int) -> Tuple[torch.Tensor, ...]:
        img_path, mask_path, flow_path = self.all_samples[idx]
        
        class_name = self._extract_class_name(img_path)

        img = self._load_image(img_path, 'RGB')
        mask = self._load_image(mask_path, 'L')
        flow = self._load_image(flow_path, 'RGB')
 
        img, mask, flow = self.augmenter.flip_horizontal(img, mask, flow)
        img, mask, flow = self.augmenter.random_crop(img, mask, flow)
        img, mask, flow = self.augmenter.rotate_random(img, mask, flow)
        img = self.augmenter.adjust_colors(img)

        img_tensor = self.img_transform(img)
        mask_tensor = self.mask_transform(mask)
        flow_tensor = self.flow_transform(flow)

        ref_img_paths, ref_flow_paths = self._get_reference_frames(class_name, num_refs=4)
        ref_imgs, ref_flows = self._process_reference_frames(ref_img_paths, ref_flow_paths)
        
        return img_tensor, mask_tensor, flow_tensor, ref_imgs, ref_flows
    
    def __len__(self) -> int:
        return len(self.all_samples)


class VideoSequenceTestDataset(data.Dataset):
    
    def __init__(self, dataset_name: str, img_size: int = None):
        self.img_size = img_size or config.TRAIN['img_size']
        self.data_root = Path(config.DATA['data_root'])
        self.dataset_name = dataset_name

        self.samples = self._load_all_sequences()
        self.class_index = self._build_class_index()

        self.img_transform = transforms.Compose([
            transforms.Resize((self.img_size, self.img_size)),
            transforms.ToTensor(),
            transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])])
        
        self.mask_transform = transforms.Compose([
            transforms.Resize((self.img_size, self.img_size)),
            transforms.ToTensor()])
        
        self.flow_transform = transforms.Compose([
            transforms.Resize((self.img_size, self.img_size)),
            transforms.ToTensor()])
    
    def _load_all_sequences(self) -> List[Tuple[str, str, str, str]]:
        all_samples = []
        base_path = self.data_root / self.dataset_name
        
        for seq_dir in sorted(base_path.iterdir()):
            if not seq_dir.is_dir():
                continue
            
            img_dir = seq_dir / 'RGB'
            mask_dir = seq_dir / 'GT'
            flow_dir = seq_dir / 'FLOW'
            
            if not all([img_dir.exists(), mask_dir.exists(), flow_dir.exists()]):
                continue

            img_files = sorted(img_dir.glob('*.jpg'))
            mask_files = sorted(list(mask_dir.glob('*.jpg')) + list(mask_dir.glob('*.png')))
            flow_files = sorted(list(flow_dir.glob('*.jpg')) + list(flow_dir.glob('*.png')))

            for img, mask, flow in zip(img_files, mask_files, flow_files):
                all_samples.append((str(img), str(mask), str(flow), seq_dir.name))
        
        return all_samples
    
    def _build_class_index(self) -> Dict[str, Dict[str, List[str]]]:
        class_index = {}
        
        for img_path, _, flow_path, _ in self.samples:
            class_name = self._extract_class_name(img_path)
            
            if class_name not in class_index:
                class_index[class_name] = {'images': [], 'flows': []}
            
            class_index[class_name]['images'].append(img_path)
            class_index[class_name]['flows'].append(flow_path)
        
        return class_index
    
    @staticmethod
    def _extract_class_name(filepath: str) -> str:
        filename = Path(filepath).stem
        return filename.split('_')[0]
    
    def _load_image(self, path: str, mode: str = 'RGB') -> Image.Image:
        with open(path, 'rb') as f:
            return Image.open(f).convert(mode)
    
    def _get_evenly_spaced_references(self, class_name: str, num_refs: int = 4) -> Tuple[List, List]:
        class_data = self.class_index[class_name]
        ref_imgs = class_data['images']
        ref_flows = class_data['flows']
        
        total_frames = len(ref_imgs)
        spacing = total_frames // num_refs
        
        sampled_imgs = [ref_imgs[i * spacing] for i in range(num_refs)]
        sampled_flows = [ref_flows[i * spacing] for i in range(num_refs)]
        
        return sampled_imgs, sampled_flows
    
    def _process_reference_frames(self, ref_img_paths: List[str], ref_flow_paths: List[str]) -> Tuple[torch.Tensor, torch.Tensor]:
        ref_img_tensors = []
        ref_flow_tensors = []
        
        for img_path, flow_path in zip(ref_img_paths, ref_flow_paths):
            img = self._load_image(img_path, 'RGB')
            flow = self._load_image(flow_path, 'RGB')
            
            img_tensor = self.img_transform(img)
            flow_tensor = self.flow_transform(flow)
            
            ref_img_tensors.append(img_tensor)
            ref_flow_tensors.append(flow_tensor)
        ref_imgs = torch.cat(ref_img_tensors, dim=0)
        ref_flows = torch.cat(ref_flow_tensors, dim=0)
        
        return ref_imgs, ref_flows
    
    def __getitem__(self, idx: int) -> Tuple:
        img_path, mask_path, flow_path, seq_name = self.samples[idx]

        class_name = self._extract_class_name(img_path)

        img = self._load_image(img_path, 'RGB')
        mask = self._load_image(mask_path, 'L')
        flow = self._load_image(flow_path, 'RGB')
        orig_size = mask.size
        img_tensor = self.img_transform(img)
        flow_tensor = self.flow_transform(flow)
        ref_img_paths, ref_flow_paths = self._get_evenly_spaced_references(class_name, num_refs=4)
        ref_imgs, ref_flows = self._process_reference_frames(ref_img_paths, ref_flow_paths)
        filename = Path(img_path).stem + '.png'
        img_resized = img.resize((self.img_size, self.img_size))
        img_resized_arr = np.array(img_resized)
        
        metadata = [orig_size, seq_name, filename]
        mask_tensor = self.mask_transform(mask)
        
        return img_tensor, mask_tensor, flow_tensor, metadata, img_resized_arr, ref_imgs, ref_flows
    
    def __len__(self) -> int:
        return len(self.samples)


def create_video_train_loader(shuffle=True, num_workers=12, pin_memory=False, drop_last=True):
    dataset = VideoSequenceDataset()
    return data.DataLoader(
        dataset=dataset,
        batch_size=config.TRAIN['batch_size'],
        shuffle=shuffle,
        num_workers=num_workers,
        pin_memory=pin_memory,
        drop_last=drop_last)


def create_video_test_loader(dataset_name, shuffle=False, num_workers=12, pin_memory=False):
    dataset = VideoSequenceTestDataset(dataset_name)
    return data.DataLoader(
        dataset=dataset,
        batch_size=config.TRAIN['batch_size'],
        shuffle=shuffle,
        num_workers=num_workers,
        pin_memory=pin_memory)

get_loader = create_video_train_loader
get_testloader = create_video_test_loader