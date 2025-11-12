
import numpy as np
import cv2
import warnings
from typing import Tuple, Optional


class SegmentationMetrics:
    @staticmethod
    def compute_iou(prediction: np.ndarray, ground_truth: np.ndarray):
        pred_binary = prediction.astype(bool)
        gt_binary = ground_truth.astype(bool)
        valid = np.ones_like(pred_binary, dtype=bool)

        inter = np.sum((pred_binary & gt_binary) & valid, axis=(-2, -1))
        union = np.sum((pred_binary | gt_binary) & valid, axis=(-2, -1))

        with np.errstate(divide="ignore", invalid="ignore"):
            iou = np.where(union == 0, 1.0, inter / union)

        return iou

    @staticmethod
    def compute_boundary_f_measure(prediction: np.ndarray, ground_truth: np.ndarray, boundary_threshold: float = 0.008):
        if ground_truth.ndim == 3:
            # [T, H, W]
            T = ground_truth.shape[0]
            scores = np.zeros(T, dtype=np.float32)
            for t in range(T):
                scores[t] = SegmentationMetrics._compute_frame_f_measure(prediction[t], ground_truth[t], boundary_threshold)
            return scores
        elif ground_truth.ndim == 2:
            return SegmentationMetrics._compute_frame_f_measure(prediction, ground_truth, boundary_threshold)
        else:
            raise ValueError("ground_truth must be 2D or 3D (T,H,W)")

    @staticmethod
    def _compute_frame_f_measure(pred_mask: np.ndarray, gt_mask: np.ndarray, boundary_threshold: float = 0.008):
        pred = (pred_mask > 0).astype(np.uint8)
        gt = (gt_mask > 0).astype(np.uint8)
        if boundary_threshold >= 1:
            radius = int(boundary_threshold)
        else:
            diag = (pred.shape[0] ** 2 + pred.shape[1] ** 2) ** 0.5
            radius = max(1, int(np.ceil(boundary_threshold * diag)))
        pb = SegmentationMetrics._extract_boundary(pred)
        gb = SegmentationMetrics._extract_boundary(gt)

        try:
            from skimage.morphology import disk
            kernel = disk(radius).astype(np.uint8)
        except Exception:
            k = 2 * radius + 1
            kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (k, k))

        pb_d = cv2.dilate(pb.astype(np.uint8), kernel)
        gb_d = cv2.dilate(gb.astype(np.uint8), kernel)
        gt_match = (gb > 0) & (pb_d > 0)
        pred_match = (pb > 0) & (gb_d > 0)
        n_pred = int(pb.sum())
        n_gt = int(gb.sum())

        if n_pred == 0 and n_gt > 0:
            precision, recall = 1.0, 0.0
        elif n_pred > 0 and n_gt == 0:
            precision, recall = 0.0, 1.0
        elif n_pred == 0 and n_gt == 0:
            precision, recall = 1.0, 1.0
        else:
            precision = float(pred_match.sum()) / float(n_pred)
            recall = float(gt_match.sum()) / float(n_gt)

        if precision + recall == 0:
            return 0.0
        return 2.0 * precision * recall / (precision + recall)

    @staticmethod
    def _extract_boundary(segmentation_mask: np.ndarray, target_width: Optional[int] = None, target_height: Optional[int] = None):
        m = (segmentation_mask > 0).astype(np.uint8)
        east = np.zeros_like(m)
        south = np.zeros_like(m)
        se = np.zeros_like(m)
        east[:, :-1] = m[:, 1:]
        south[:-1, :] = m[1:, :]
        se[:-1, :-1] = m[1:, 1:]

        boundary = ((m ^ east) | (m ^ south) | (m ^ se)).astype(np.uint8)
        boundary[-1, :] = m[-1, :] ^ east[-1, :]
        boundary[:, -1] = m[:, -1] ^ south[:, -1]
        boundary[-1, -1] = 0

        if target_width is None and target_height is None:
            return boundary
        if target_width is None or target_height is None:
            raise ValueError("Provide both target_width and target_height, or neither.")

        return cv2.resize(boundary, (target_width, target_height), interpolation=cv2.INTER_NEAREST)

    @staticmethod
    def compute_statistics(frame_values: np.ndarray):
        frame_values = np.asarray(frame_values, dtype=np.float32)
        with warnings.catch_warnings():
            warnings.simplefilter('ignore', category=RuntimeWarning)
            mean_score = np.nanmean(frame_values)
            recall_score = np.nanmean(frame_values > 0.5)
        n = len(frame_values)
        if n == 0:
            return float('nan'), float('nan'), float('nan')
        edges = np.linspace(0, n, 5, dtype=int)  # 0..n into 4 bins
        bins = [frame_values[edges[i]:edges[i + 1]] for i in range(4)]

        with warnings.catch_warnings():
            warnings.simplefilter('ignore', category=RuntimeWarning)
            decay_score = float(np.nanmean(bins[0]) - np.nanmean(bins[3]))

        return float(mean_score), float(recall_score), float(decay_score)

def db_eval_iou(annotation, segmentation):
    return SegmentationMetrics.compute_iou(segmentation, annotation)


def db_eval_boundary(annotation, segmentation, bound_th=0.008):
    return SegmentationMetrics.compute_boundary_f_measure(segmentation, annotation, bound_th)


def f_measure(foreground_mask, gt_mask, bound_th=0.008):
    return SegmentationMetrics._compute_frame_f_measure(foreground_mask, gt_mask, bound_th)


def _seg2bmap(seg, width=None, height=None):
    return SegmentationMetrics._extract_boundary(seg, width, height)


def db_statistics(per_frame_values):
    return SegmentationMetrics.compute_statistics(per_frame_values)
