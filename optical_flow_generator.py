import sys
sys.path.append('RAFT/core')

import os
import cv2
import torch
import numpy as np
from PIL import Image
from pathlib import Path
import argparse
from glob import glob
import shutil

from RAFT import RAFT
from utils.utils import InputPadder

DAVIS_TRAIN_SEQUENCES = [
    'bear', 'bmx-bumps', 'boat', 'breakdance', 'bus', 'car-turn', 
    'dance-jump', 'dog-agility', 'drift-turn', 'elephant', 'flamingo', 
    'hike', 'hockey', 'horsejump-low', 'kite-walk', 'lucia', 
    'mallard-fly', 'mallard-water', 'motocross-bumps', 'motorbike', 
    'paragliding', 'parkour', 'scooter-gray', 'soccerball', 'stroller', 
    'surf', 'swing', 'tennis', 'train'
]

DAVIS_VAL_SEQUENCES = [
    'blackswan', 'bmx-trees', 'breakdance-flare', 'camel', 'car-roundabout', 
    'car-shadow', 'cows', 'dance-twirl', 'dog', 'drift-chicane', 
    'drift-straight', 'goat', 'horsejump-high', 'kite-surf', 'libby', 
    'motocross-jump', 'paragliding-launch', 'soapbox'
]

def load_image(img_path):
    img = np.array(Image.open(img_path)).astype(np.uint8)
    if len(img.shape) == 2:
        img = np.stack([img] * 3, axis=-1)
    img = torch.from_numpy(img).permute(2, 0, 1).float()
    return img[None].to('cuda')

def flow_to_image(flow):
    hsv = np.zeros((flow.shape[0], flow.shape[1], 3), dtype=np.uint8)
    hsv[..., 1] = 255
    
    mag, ang = cv2.cartToPolar(flow[..., 0], flow[..., 1])
    hsv[..., 0] = ang * 180 / np.pi / 2
    hsv[..., 2] = cv2.normalize(mag, None, 0, 255, cv2.NORM_MINMAX)
    
    rgb = cv2.cvtColor(hsv, cv2.COLOR_HSV2BGR)
    return rgb

def generate_flow_for_sequence(model, rgb_dir, flow_dir):
    image_files = sorted(glob(os.path.join(rgb_dir, '*.jpg')))
    if not image_files:
        image_files = sorted(glob(os.path.join(rgb_dir, '*.png')))
    
    if len(image_files) == 0:
        print(f"No images found in {rgb_dir}")
        return
    
    os.makedirs(flow_dir, exist_ok=True)
    
    for i in range(len(image_files) - 1):
        img1_path = image_files[i]
        img2_path = image_files[i + 1]
        
        image1 = load_image(img1_path)
        image2 = load_image(img2_path)
        
        padder = InputPadder(image1.shape)
        image1, image2 = padder.pad(image1, image2)
        
        with torch.no_grad():
            _, flow = model(image1, image2, iters=20, test_mode=True)
        
        flow = padder.unpad(flow[0]).cpu().numpy().transpose(1, 2, 0)
        
        flow_img = flow_to_image(flow)
        
        frame_name = os.path.basename(img1_path)
        frame_name = os.path.splitext(frame_name)[0] + '.jpg'
        flow_path = os.path.join(flow_dir, frame_name)
        cv2.imwrite(flow_path, flow_img)
        
        if (i + 1) % 10 == 0:
            print(f"  Processed {i + 1}/{len(image_files)-1} flow maps")
    
    last_frame_name = os.path.splitext(os.path.basename(image_files[-1]))[0] + '.jpg'
    last_flow_path = os.path.join(flow_dir, last_frame_name)
    second_last_flow = os.path.join(flow_dir, 
                                    os.path.splitext(os.path.basename(image_files[-2]))[0] + '.jpg')
    if os.path.exists(second_last_flow):
        shutil.copy(second_last_flow, last_flow_path)
    
    return len(image_files)

def organize_davis_dataset(davis_root, output_dir, resolution='480p'):
    jpeg_dir = os.path.join(davis_root, 'JPEGImages', resolution)
    annot_dir = os.path.join(davis_root, 'Annotations', resolution)
    
    if not os.path.exists(jpeg_dir):
        print(f"Error: {jpeg_dir} does not exist!")
        return
    
    train_rgb_dir = os.path.join(output_dir, 'DAVIS_train', 'RGB')
    train_gt_dir = os.path.join(output_dir, 'DAVIS_train', 'GT')
    train_flow_dir = os.path.join(output_dir, 'DAVIS_train', 'FLOW')
    
    os.makedirs(train_rgb_dir, exist_ok=True)
    os.makedirs(train_gt_dir, exist_ok=True)
    os.makedirs(train_flow_dir, exist_ok=True)
    
    print(f"\nOrganizing training sequences...")
    train_frame_count = 0
    for seq in DAVIS_TRAIN_SEQUENCES:
        seq_jpeg = os.path.join(jpeg_dir, seq)
        seq_annot = os.path.join(annot_dir, seq)
        
        if not os.path.exists(seq_jpeg):
            print(f"  Warning: {seq} not found, skipping...")
            continue
        
        print(f"  Processing {seq}...")
        
        rgb_files = sorted(glob(os.path.join(seq_jpeg, '*.*')))
        for rgb_file in rgb_files:
            frame_name = os.path.basename(rgb_file)
            dst_name = f"{seq}_{frame_name}"
            shutil.copy(rgb_file, os.path.join(train_rgb_dir, dst_name))
        
        if os.path.exists(seq_annot):
            gt_files = sorted(glob(os.path.join(seq_annot, '*.png')))
            for gt_file in gt_files:
                frame_name = os.path.basename(gt_file)
                dst_name = f"{seq}_{frame_name}"
                shutil.copy(gt_file, os.path.join(train_gt_dir, dst_name))
        
        train_frame_count += len(rgb_files)
    
    print(f"  Total training frames: {train_frame_count}")
    
    print(f"\nOrganizing test sequences...")
    test_base_dir = os.path.join(output_dir, 'DAVIS_test')
    
    for seq in DAVIS_VAL_SEQUENCES:
        seq_jpeg = os.path.join(jpeg_dir, seq)
        seq_annot = os.path.join(annot_dir, seq)
        
        if not os.path.exists(seq_jpeg):
            print(f"  Warning: {seq} not found, skipping...")
            continue
        
        print(f"  Processing {seq}...")
        
        seq_out_dir = os.path.join(test_base_dir, seq)
        seq_rgb_dir = os.path.join(seq_out_dir, 'RGB')
        seq_gt_dir = os.path.join(seq_out_dir, 'GT')
        seq_flow_dir = os.path.join(seq_out_dir, 'FLOW')
        
        os.makedirs(seq_rgb_dir, exist_ok=True)
        os.makedirs(seq_gt_dir, exist_ok=True)
        os.makedirs(seq_flow_dir, exist_ok=True)
        
        rgb_files = sorted(glob(os.path.join(seq_jpeg, '*.*')))
        for rgb_file in rgb_files:
            frame_name = os.path.basename(rgb_file)
            dst_name = f"{seq}_{frame_name}"
            shutil.copy(rgb_file, os.path.join(seq_rgb_dir, dst_name))
        
        if os.path.exists(seq_annot):
            gt_files = sorted(glob(os.path.join(seq_annot, '*.png')))
            for gt_file in gt_files:
                frame_name = os.path.basename(gt_file)
                dst_name = f"{seq}_{frame_name}"
                shutil.copy(gt_file, os.path.join(seq_gt_dir, dst_name))
    
    print(f"\nDataset organization complete!")
    print(f"Training sequences: {len(DAVIS_TRAIN_SEQUENCES)}")
    print(f"Test sequences: {len(DAVIS_VAL_SEQUENCES)}")

def process_davis_flow(model, dataset_dir, mode='both'):
    if mode in ['train', 'both']:
        print("\n" + "="*70)
        print("Processing DAVIS_train")
        print("="*70)
        rgb_dir = os.path.join(dataset_dir, 'DAVIS_train', 'RGB')
        flow_dir = os.path.join(dataset_dir, 'DAVIS_train', 'FLOW')
        
        if os.path.exists(rgb_dir):
            total_frames = generate_flow_for_sequence(model, rgb_dir, flow_dir)
            print(f"Completed! Generated flow for {total_frames} frames")
        else:
            print(f"Error: {rgb_dir} does not exist!")
    
    if mode in ['test', 'both']:
        print("\n" + "="*70)
        print("Processing DAVIS_test")
        print("="*70)
        davis_test_dir = os.path.join(dataset_dir, 'DAVIS_test')
        
        if not os.path.exists(davis_test_dir):
            print(f"Error: {davis_test_dir} does not exist!")
            return
        
        sequences = [d for d in os.listdir(davis_test_dir) 
                    if os.path.isdir(os.path.join(davis_test_dir, d))]
        
        for idx, seq in enumerate(sorted(sequences), 1):
            print(f"\n[{idx}/{len(sequences)}] Processing sequence: {seq}")
            rgb_dir = os.path.join(davis_test_dir, seq, 'RGB')
            flow_dir = os.path.join(davis_test_dir, seq, 'FLOW')
            
            if os.path.exists(rgb_dir):
                total_frames = generate_flow_for_sequence(model, rgb_dir, flow_dir)
                print(f"  Completed {seq}! Generated {total_frames} flow maps")
            else:
                print(f"  Warning: RGB directory not found for {seq}")

def main(args):
    if args.organize:
        print("="*70)
        print("STEP 1: Organizing DAVIS dataset")
        print("="*70)
        organize_davis_dataset(args.davis_root, args.output_dir, args.resolution)
        dataset_dir = args.output_dir
    else:
        dataset_dir = args.dataset_dir
    
    if args.generate_flow:
        print("\n" + "="*70)
        print("STEP 2: Generating optical flow with RAFT")
        print("="*70)
        print("Loading RAFT model...")
        model = torch.nn.DataParallel(RAFT(args))
        model.load_state_dict(torch.load(args.model))
        model = model.module
        model.to('cuda')
        model.eval()
        print("RAFT model loaded successfully!\n")
        
        process_davis_flow(model, dataset_dir, args.mode)
        
        print("\n" + "="*70)
        print("Optical flow generation completed!")
        print("="*70)

if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='DAVIS dataset preparation and flow generation')
    
    parser.add_argument('--organize', action='store_true',
                       help='Organize DAVIS dataset into train/test structure')
    parser.add_argument('--davis_root', type=str,
                       help='Path to original DAVIS dataset root')
    parser.add_argument('--output_dir', type=str,
                       help='Path to output organized dataset')
    parser.add_argument('--resolution', type=str, default='480p',
                       choices=['480p', '1080p'],
                       help='Resolution to use (480p or 1080p)')
    
    parser.add_argument('--generate_flow', action='store_true',
                       help='Generate optical flow maps')
    parser.add_argument('--dataset_dir', type=str,
                       help='Path to organized dataset directory')
    parser.add_argument('--model', type=str, default='models/raft-things.pth',
                       help='Path to RAFT model checkpoint')
    parser.add_argument('--mode', type=str, default='both',
                       choices=['train', 'test', 'both'],
                       help='Process train, test, or both')
    
    parser.add_argument('--small', action='store_true',
                       help='Use small RAFT model')
    parser.add_argument('--mixed_precision', action='store_true',
                       help='Use mixed precision')
    parser.add_argument('--alternate_corr', action='store_true',
                       help='Use alternate correlation implementation')
    
    args = parser.parse_args()
    
    if not args.organize and not args.generate_flow:
        parser.error("Must specify --organize and/or --generate_flow")
    
    if args.organize and not (args.davis_root and args.output_dir):
        parser.error("--organize requires --davis_root and --output_dir")
    
    if args.generate_flow and not args.dataset_dir:
        parser.error("--generate_flow requires --dataset_dir")
    
    main(args)