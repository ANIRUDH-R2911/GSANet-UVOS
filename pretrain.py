import os
from pathlib import Path
import numpy as np
import cv2
import random
import torch
from torch.cuda.amp import autocast, GradScaler
import torch.nn.functional as F
from dataloader.data_for_pretrain import get_loader, get_testloader
from loss import structure_loss
from model.model_for_pretrain import GSANet
import config as config
from logger import *

os.environ["CUDA_VISIBLE_DEVICES"] = config.TRAIN['GPU']
BEST_METRIC = 0.0

VALIDATION_ROOT = Path(config.DATA['data_root']) / config.DATA['DAVIS_val']
VALIDATION_SEQUENCES = [d.name for d in VALIDATION_ROOT.iterdir() if d.is_dir()]


class TrainingEngine:
    def __init__(self, model, optimizer, scheduler, device):
        self.model = model
        self.optimizer = optimizer
        self.scheduler = scheduler
        self.device = device
        self.timer = pttm()
        self.scaler = GradScaler('cuda') 

    def run_epoch(self, epoch_num, data_loader):
        self.model.train()
        epoch_loss = 0.0

        for batch_idx, (images, masks) in enumerate(data_loader):
            self.timer.display_progress(epoch_num, batch_idx, data_loader)
            if images.size(0) == 1:
                continue

            train_size = self._get_training_size()
            images = self._resize_tensor(images, train_size).to(self.device)
            masks = self._resize_tensor(masks, train_size).to(self.device)

            self.optimizer.zero_grad()

            with autocast('cuda'):
                predictions, _ = self.model(images)
                batch_loss = self._compute_total_loss(predictions, masks)

            self.scaler.scale(batch_loss).backward()
            self.scaler.step(self.optimizer)
            self.scaler.update()
            self.scheduler.step()

            epoch_loss += batch_loss.item()

        avg_loss = epoch_loss / len(data_loader)
        lr = self.optimizer.param_groups[-1]['lr']
        print(f"\nEpoch: #{epoch_num}\tLr: {lr:.7f}\tLoss: {avg_loss:.4f}\n")

    def _get_training_size(self):
        if config.TRAIN['img_size'] == 384:
            return 384
        return random.choice([384, 416, 448, 480, 512])

    @staticmethod
    def _resize_tensor(tensor, size):
        return F.interpolate(tensor, size=(size, size), mode='bicubic', align_corners=False)

    @staticmethod
    def _compute_total_loss(predictions, ground_truth):
        total_loss = 0
        for pred in predictions:
            total_loss += structure_loss(pred, ground_truth)
        return total_loss


class ValidationEngine:
    def __init__(self, model, device, sequence_names):
        self.model = model
        self.device = device
        self.sequence_names = sequence_names
        self.timer = pttm()

    def evaluate(self, epoch_num, workspace_dir):
        global BEST_METRIC
        print("Evaluating model...")

        val_loader = get_testloader()
        jaccard_scores = {seq: [] for seq in self.sequence_names}
        self.model.eval()

        with torch.no_grad():
            for batch_idx, batch_data in enumerate(val_loader):
                self.timer.display_progress(epoch_num, batch_idx, val_loader)

                image, gt, metadata, _ = batch_data
                batch_size = image.shape[0]
                orig_heights = metadata[0][0]
                orig_widths = metadata[0][1]

                image, gt = image.to(self.device), gt.to(self.device)
                predictions, _ = self.model(image)
                pred_map = predictions[0]

                for b in range(batch_size):
                    score = self._compute_jaccard_sample(
                        pred_map[b], gt[b],
                        orig_heights[b].item(),
                        orig_widths[b].item())
                    sequence_name = metadata[1][b]
                    jaccard_scores[sequence_name].append(score)

        mean_jaccard = self._compute_mean_jaccard(jaccard_scores)
        print(f"Mean Jaccard Index: {mean_jaccard:.4f}")

        if mean_jaccard > BEST_METRIC:
            BEST_METRIC = mean_jaccard
            save_model(workspace_dir, epoch_num, self.model, 'best')
            print("Saved best model!")
            return True
        return False

    def _compute_jaccard_sample(self, pred, gt, height, width):
        pred = pred.unsqueeze(0).float()
        gt = gt.unsqueeze(0).float()
        pred = F.interpolate(pred, size=(height, width), mode='bilinear', align_corners=False)
        gt = F.interpolate(gt, size=(height, width), mode='bilinear', align_corners=False)

        pred = (pred - pred.min()) / (pred.max() - pred.min() + 1e-8)
        pred = (pred > 0.5).float()
        gt = gt / (torch.max(gt) + 1e-8)

        intersection = torch.sum(pred * gt)
        union = torch.sum(pred) + torch.sum(gt) - intersection
        jaccard = (intersection / union).item() if union > 0 else 0
        return jaccard

    @staticmethod
    def _compute_mean_jaccard(scores_dict):
        return np.mean([np.mean(scores) for scores in scores_dict.values()])


class VisualizationEngine:
    def __init__(self, model, device, workspace_dir):
        self.model = model
        self.device = device
        self.workspace_dir = Path(workspace_dir)

    def generate_visualizations(self):
        val_loader = get_testloader()
        self.model.eval()

        with torch.no_grad():
            for batch_data in val_loader:
                image, gt, metadata, img_raw = batch_data
                batch_size = image.shape[0]
                orig_heights = metadata[0][0]
                orig_widths = metadata[0][1]

                image = image.to(self.device)
                predictions, _ = self.model(image)
                pred_map = predictions[0]

                for b in range(batch_size):
                    self._save_sample_visualization(
                        pred_map[b], gt[b], img_raw[b],
                        metadata[1][b], metadata[2][b],
                        orig_heights[b].item(), orig_widths[b].item())

    def _save_sample_visualization(self, pred, gt, raw_img, seq_name, filename, height, width):
        pred_np = self._normalize_array(pred.squeeze().cpu().numpy())
        gt_np = gt.squeeze().cpu().numpy() / (gt.max() + 1e-8)
        raw_np = raw_img.squeeze().numpy()

        pred_vis = self._to_bgr(pred_np, height, width)
        gt_vis = self._to_bgr(gt_np, height, width)
        raw_vis = cv2.resize(cv2.cvtColor(raw_np, cv2.COLOR_RGB2BGR), (width, height))

        combined = cv2.hconcat([raw_vis, pred_vis, gt_vis])
        self._save_results(combined, pred_vis, gt_vis, seq_name, filename)

    @staticmethod
    def _normalize_array(arr):
        return (arr - arr.min()) / (arr.max() - arr.min() + 1e-8)

    @staticmethod
    def _to_bgr(arr, h, w):
        img = cv2.cvtColor((arr * 255).astype(np.uint8), cv2.COLOR_GRAY2BGR)
        return cv2.resize(img, (w, h), interpolation=cv2.INTER_AREA)

    def _save_results(self, combined, pred, gt, seq_name, filename):
        total_dir = self.workspace_dir / "result" / "total" / seq_name
        pred_dir = self.workspace_dir / "result" / "pred" / seq_name
        gt_dir = self.workspace_dir / "result" / "gt" / seq_name

        for d in [total_dir, pred_dir, gt_dir]:
            d.mkdir(parents=True, exist_ok=True)

        cv2.imwrite(str(total_dir / filename), combined)
        cv2.imwrite(str(pred_dir / filename), pred)
        cv2.imwrite(str(gt_dir / filename), gt)

def load_pretrained_model(workspace_dir, device):
    model = GSANet()
    model = torch.nn.DataParallel(model.to(device))
    checkpoint = torch.load(Path(workspace_dir) / "model" / "best_model.pth")
    model.load_state_dict(checkpoint['model_state_dict'])
    return model


def initialize_training_components():
    workspace = make_new_work_space()
    save_config_file(workspace)

    print("Loading dataset...")
    train_loader = get_loader()
    print(f"Training samples: {len(train_loader)}")

    device = torch.device("cuda")
    print(f"Using device: {device}")

    print("Building model...")
    model = torch.nn.DataParallel(GSANet().to(device))

    total_params = sum(np.prod(p.size()) for p in model.parameters())
    trainable_params = sum(np.prod(p.shape) for p in model.parameters() if p.requires_grad)
    print(f"Total parameters: {total_params:,}")
    print(f"Trainable parameters: {trainable_params:,}")

    optimizer = torch.optim.Adam(model.parameters(), lr=config.TRAIN['learning_rate'])
    total_steps = len(train_loader) * config.TRAIN['epoch']
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
        optimizer, T_max=total_steps, eta_min=config.TRAIN['learning_rate'] / 10)

    return workspace, train_loader, model, optimizer, scheduler, device


def run_training_pipeline():
    workspace, train_loader, model, optimizer, scheduler, device = initialize_training_components()
    trainer = TrainingEngine(model, optimizer, scheduler, device)
    validator = ValidationEngine(model, device, VALIDATION_SEQUENCES)

    print("\n" + "=" * 50)
    print("Starting training...")
    print("=" * 50 + "\n")

    for epoch in range(config.TRAIN['epoch']):
        trainer.run_epoch(epoch, train_loader)
        is_best = validator.evaluate(epoch, workspace)
        if is_best:
            visualizer = VisualizationEngine(
                load_pretrained_model(workspace, device), device, workspace)
            visualizer.generate_visualizations()
        print("")

    print("=" * 50)
    print("Training completed!")
    print("=" * 50)


if __name__ == "__main__":
    run_training_pipeline()
