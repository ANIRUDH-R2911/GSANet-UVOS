from datetime import datetime
from pathlib import Path
import os
import shutil
import torch
import time
from typing import Optional


class ProgressTracker:
    
    def __init__(self):
        self.start_time = time.time()
        self.last_update_time = self.start_time
    
    def display_progress(self, epoch_num, batch_idx, data_loader):
        total_batches = len(data_loader)
        current_batch = batch_idx + 1
        current_time = time.time()
        
        batch_duration = current_time - self.last_update_time
        self.last_update_time = current_time
        
        batches_remaining = total_batches - current_batch
        estimated_remaining = int(batches_remaining * batch_duration)
        remaining_str = time.strftime('%H:%M:%S', time.gmtime(estimated_remaining))
        
        elapsed_seconds = int(current_time - self.start_time)
        elapsed_str = time.strftime('%H:%M:%S', time.gmtime(elapsed_seconds))
        
        progress_pct = int((current_batch / total_batches) * 100)
        
        bar_width = 30
        filled_width = int((current_batch / total_batches) * bar_width)
        empty_width = bar_width - filled_width
        
        filled_bar = "\033[101m" + " " * filled_width + "\033[0m" 
        empty_bar = "\033[43m" + " " * empty_width + "\033[0m"     
        
        status = (f"Epoch: {epoch_num} [{current_batch}/{total_batches} ({progress_pct}%)] "
                 f"[{elapsed_str}/{remaining_str}]   {filled_bar}{empty_bar}")
        
        print(status, end="\r")
    
    def print_status(self, epoch_num, batch_idx, data_loader):
          self.display_progress(epoch_num, batch_idx, data_loader)

    def reset(self):
        self.start_time = time.time()
        self.last_update_time = self.start_time


class TimestampGenerator:    
    def __init__(self, timezone_name='Asia/Seoul'):
        try:
            from pytz import timezone
            self.timezone = timezone(timezone_name)
        except ImportError:
            self.timezone = None
    
    def get_timestamp(self, format_str='%Y-%m-%d %H:%M:%S'):
        if self.timezone:
            return datetime.now(self.timezone).strftime(format_str)
        else:
            return datetime.now().strftime(format_str)


class WorkspaceManager:    
    def __init__(self, base_dir='./log'):
        self.base_dir = Path(base_dir)
        self.timestamp_gen = TimestampGenerator()
    
    def create_workspace(self):
        timestamp = self.timestamp_gen.get_timestamp()
        workspace_path = self.base_dir / timestamp
        workspace_path.mkdir(parents=True, exist_ok=True)
        
        return str(workspace_path)
    
    def save_checkpoint(self, workspace_dir, epoch_num, model, checkpoint_name):
        checkpoint_dir = Path(workspace_dir) / "model"
        checkpoint_dir.mkdir(parents=True, exist_ok=True)
        
        checkpoint_path = checkpoint_dir / f"{checkpoint_name}_model.pth"
        
        torch.save({
            "epoch": epoch_num,
            "model_state_dict": model.state_dict()
        }, str(checkpoint_path))
    
    def save_config(self, workspace_dir, config_file_path='./config.py'):
        config_dir = Path(workspace_dir) / "train"
        config_dir.mkdir(parents=True, exist_ok=True)
        
        destination = config_dir / "config.py"
        
        if not destination.exists():
            shutil.copy(config_file_path, str(destination))
    
    def append_test_log(self, workspace_dir, message):
        log_dir = Path(workspace_dir) / "test"
        log_dir.mkdir(parents=True, exist_ok=True)
        
        log_file = log_dir / "test_log.txt"
        
        with open(str(log_file), 'a') as f:
            f.write(message + "\n")
    
    def copy_results(self, workspace_dir, sequence_name):
        workspace_path = Path(workspace_dir)
        subdirs = ['total', 'pred', 'gt']
        
        for subdir in subdirs:
            source_dir = workspace_path / "buffer" / subdir / sequence_name
            dest_dir = workspace_path / "result" / subdir / sequence_name
            
            if not source_dir.exists():
                continue
            
            for root, _, files in os.walk(str(source_dir)):
                root_path = Path(root)
                
                for filename in files:
                    if not filename.endswith('.png'):
                        continue
                    
                    relative_path = root_path.relative_to(source_dir)
                    dest_path = dest_dir / relative_path
                    dest_path.mkdir(parents=True, exist_ok=True)
                    
                    source_file = root_path / filename
                    dest_file = dest_path / filename
                    shutil.copy(str(source_file), str(dest_file))


class ExperimentLogger:    
    def __init__(self, base_log_dir='./log', timezone_name='America/New_York'):
        self.workspace_mgr = WorkspaceManager(base_log_dir)
        self.progress_tracker = ProgressTracker()
        self.timestamp_gen = TimestampGenerator(timezone_name)
        self.current_workspace = None
    
    def start_experiment(self):
        self.current_workspace = self.workspace_mgr.create_workspace()
        return self.current_workspace
    
    def log_progress(self, epoch, batch_idx, dataloader):
        self.progress_tracker.display_progress(epoch, batch_idx, dataloader)
    
    def save_model(self, epoch, model, name='checkpoint'):
        if self.current_workspace:
            self.workspace_mgr.save_checkpoint(
                self.current_workspace, 
                epoch, 
                model, 
                name)
    
    def save_config(self, config_path='./config.py'):
        if self.current_workspace:
            self.workspace_mgr.save_config(self.current_workspace, config_path)
    
    def log_test_result(self, message):
        if self.current_workspace:
            self.workspace_mgr.append_test_log(self.current_workspace, message)
    
    def finalize_results(self, sequence_name):
        if self.current_workspace:
            self.workspace_mgr.copy_results(self.current_workspace, sequence_name)

def get_current_time():
    timestamp_gen = TimestampGenerator()
    return timestamp_gen.get_timestamp()


def make_new_work_space():
    workspace_mgr = WorkspaceManager()
    return workspace_mgr.create_workspace()


def save_model(root_dir, epoch, model, name):
    workspace_mgr = WorkspaceManager()
    workspace_mgr.save_checkpoint(root_dir, epoch, model, name)


def save_config_file(root_dir):
    workspace_mgr = WorkspaceManager()
    workspace_mgr.save_config(root_dir)


def save_testing_log(root_dir, msg):
    workspace_mgr = WorkspaceManager()
    workspace_mgr.append_test_log(root_dir, msg)


def copy_result(work_dir, valid_name):
    workspace_mgr = WorkspaceManager()
    workspace_mgr.copy_results(work_dir, valid_name)


pttm = ProgressTracker