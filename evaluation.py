import os
import numpy as np
from PIL import Image
from tqdm import tqdm
import cv2

GT_ROOT = '/content/CV_Project/DAVIS-data/dataset/DAVIS_test'
PRED_ROOT = '/content/CV_Project/Code-Files/DAVIS-evaluation/DAVIS/Results/480p/GSANet_FT'

SEQUENCES = [
    'blackswan', 'bmx-trees', 'breakdance-flare', 'camel', 
    'car-roundabout', 'car-shadow', 'cows', 'dance-twirl', 
    'dog', 'drift-chicane', 'drift-straight', 'goat', 
    'horsejump-high', 'kite-surf', 'libby', 'motocross-jump', 
    'paragliding-launch', 'soapbox']


def db_eval_iou(annotation, segmentation):
    annotation = annotation.astype(bool)
    segmentation = segmentation.astype(bool)
    
    if np.sum(annotation) == 0 and np.sum(segmentation) == 0:
        return 1.0
    
    intersection = np.sum(annotation & segmentation)
    union = np.sum(annotation | segmentation)
    
    return intersection / union if union > 0 else 0.0


def db_eval_boundary(annotation, segmentation, bound_th=0.008):
    annotation = annotation.astype(np.uint8)
    segmentation = segmentation.astype(np.uint8)
    
    diag = np.sqrt(annotation.shape[0]**2 + annotation.shape[1]**2)
    bound_pix = int(np.ceil(bound_th * diag))
    
    kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (3, 3))
    
    gt_dilated = cv2.dilate(annotation, kernel, iterations=1)
    gt_boundary = gt_dilated - annotation
    
    pred_dilated = cv2.dilate(segmentation, kernel, iterations=1)
    pred_boundary = pred_dilated - segmentation
    
    if bound_pix > 1:
        bound_kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (bound_pix*2+1, bound_pix*2+1))
        gt_boundary_dilated = cv2.dilate(gt_boundary, bound_kernel)
        pred_boundary_dilated = cv2.dilate(pred_boundary, bound_kernel)
    else:
        gt_boundary_dilated = gt_boundary
        pred_boundary_dilated = pred_boundary
    
    gt_match = (gt_boundary > 0) & (pred_boundary_dilated > 0)
    pred_match = (pred_boundary > 0) & (gt_boundary_dilated > 0)
    
    n_gt = np.sum(gt_boundary > 0)
    n_pred = np.sum(pred_boundary > 0)
    
    if n_pred == 0 and n_gt > 0:
        precision, recall = 1.0, 0.0
    elif n_pred > 0 and n_gt == 0:
        precision, recall = 0.0, 1.0
    elif n_pred == 0 and n_gt == 0:
        precision, recall = 1.0, 1.0
    else:
        precision = np.sum(pred_match) / n_pred
        recall = np.sum(gt_match) / n_gt
    
    if precision + recall == 0:
        return 0.0
    
    return 2 * precision * recall / (precision + recall)


def load_mask(path):
    if not os.path.exists(path):
        return None
    
    mask = np.array(Image.open(path).convert('L'))
    return (mask > 127).astype(np.uint8)


def evaluate_sequence(seq_name):
    gt_dir = os.path.join(GT_ROOT, seq_name, 'GT')
    pred_dir = os.path.join(PRED_ROOT, seq_name)
    
    if not os.path.exists(pred_dir):
        print(f"Skipping {seq_name}: no predictions found")
        return None
    
    pred_frames = sorted([f for f in os.listdir(pred_dir) if f.endswith('.png')])
    
    if len(pred_frames) == 0:
        print(f"Skipping {seq_name}: no frames found")
        return None
    
    j_scores = []
    f_scores = []
    
    for frame_name in pred_frames:
        gt_path = os.path.join(gt_dir, frame_name)
        if not os.path.exists(gt_path):
            gt_path = gt_path.replace('.png', '.jpg')
        
        gt_mask = load_mask(gt_path)
        if gt_mask is None:
            continue
        pred_path = os.path.join(pred_dir, frame_name)
        pred_mask = load_mask(pred_path)
        if pred_mask is None:
            continue
        
        if pred_mask.shape != gt_mask.shape:
            pred_mask = cv2.resize(pred_mask, (gt_mask.shape[1], gt_mask.shape[0]), 
                                   interpolation=cv2.INTER_NEAREST)
        
        j = db_eval_iou(gt_mask, pred_mask)
        f = db_eval_boundary(gt_mask, pred_mask)
        
        j_scores.append(j)
        f_scores.append(f)
    
    if len(j_scores) == 0:
        return None
    
    return {
        'J_mean': np.mean(j_scores),
        'J_recall': np.mean([j > 0.5 for j in j_scores]),
        'J_decay': np.mean(j_scores[:len(j_scores)//4]) - np.mean(j_scores[-len(j_scores)//4:]) if len(j_scores) >= 4 else 0,
        'F_mean': np.mean(f_scores),
        'F_recall': np.mean([f > 0.5 for f in f_scores]),
        'F_decay': np.mean(f_scores[:len(f_scores)//4]) - np.mean(f_scores[-len(f_scores)//4:]) if len(f_scores) >= 4 else 0,
        'num_frames': len(j_scores)}


def main():
    print("-" * 60)
    print("DAVIS Evaluation")
    print("-" * 60)
    print(f"GT Root: {GT_ROOT}")
    print(f"Prediction Root: {PRED_ROOT}")
    print(f"Sequences: {len(SEQUENCES)}")
    print("=" * 60)
    
    all_results = {}
    
    for seq in tqdm(SEQUENCES, desc="Evaluating sequences"):
        result = evaluate_sequence(seq)
        if result is not None:
            all_results[seq] = result
    
    if len(all_results) == 0:
        print("\n No results computed!")
        return
    
    print("\n" + "=" * 60)
    print("Per-Sequence Results")
    print("=" * 60)
    print(f"{'Sequence':<20} {'J-Mean':>8} {'F-Mean':>8} {'J&F':>8} {'Frames':>7}")
    print("-" * 60)
    
    for seq, res in sorted(all_results.items()):
        jf_mean = (res['J_mean'] + res['F_mean']) / 2
        print(f"{seq:<20} {res['J_mean']:>8.4f} {res['F_mean']:>8.4f} {jf_mean:>8.4f} {res['num_frames']:>7}")
    
    j_means = [r['J_mean'] for r in all_results.values()]
    f_means = [r['F_mean'] for r in all_results.values()]
    j_recalls = [r['J_recall'] for r in all_results.values()]
    f_recalls = [r['F_recall'] for r in all_results.values()]
    j_decays = [r['J_decay'] for r in all_results.values()]
    f_decays = [r['F_decay'] for r in all_results.values()]
    
    overall_j = np.mean(j_means)
    overall_f = np.mean(f_means)
    overall_jf = (overall_j + overall_f) / 2
    
    print("\n" + "=" * 60)
    print("Overall Results")
    print("=" * 60)
    print(f"J&F-Mean:   {overall_jf:.4f}")
    print(f"J-Mean:     {overall_j:.4f}")
    print(f"J-Recall:   {np.mean(j_recalls):.4f}")
    print(f"J-Decay:    {np.mean(j_decays):.4f}")
    print(f"F-Mean:     {overall_f:.4f}")
    print(f"F-Recall:   {np.mean(f_recalls):.4f}")
    print(f"F-Decay:    {np.mean(f_decays):.4f}")
    print("=" * 60)
    print(f"\n Evaluated {len(all_results)} sequences successfully!")


if __name__ == "__main__":
    main()