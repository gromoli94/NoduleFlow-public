"""Evaluation for the NoduleFlow pipeline.

Given a predicted lesion mask, a ground-truth mask and the candidate detections
that prompted the segmentation, the Evaluator produces:

  * a per-candidate table  -- one row per detection, with its scores and the
    segmentation metrics (IoU / Dice vs the matched GT lesion, volume, TP/FP),
  * a global-metrics table -- detection-level (sensitivity, precision, F1) and
    segmentation-level (volumetric Dice/IoU, mean per-lesion overlap).

Connected-component and overlap helpers are transferred from the source repo
(`connected_components_3d`, `ComputeDiceAndIoU`).
"""
import numpy as np
import pandas as pd
import nibabel as nib
from scipy.ndimage import label as cc_label
from scipy.ndimage import find_objects


def connected_components_3d(mask, connectivity=26, min_size=5):
    if connectivity == 6:
        structure = np.zeros((3, 3, 3), dtype=np.uint8)
        structure[1, 1, :] = 1
        structure[1, :, 1] = 1
        structure[:, 1, 1] = 1
    elif connectivity == 26:
        structure = np.ones((3, 3, 3), dtype=np.uint8)
    else:
        raise ValueError("connectivity must be 6 or 26")
    lab, n = cc_label(mask.astype(bool), structure=structure)
    if n == 0:
        return lab.astype(np.int32), 0
    counts = np.bincount(lab.ravel())
    keep = np.zeros(n + 1, dtype=bool)
    keep[1:] = counts[1:] >= min_size
    mapping = np.zeros(n + 1, dtype=np.int32)
    mapping[keep] = np.cumsum(keep.astype(np.int32))[keep]
    lab_out = mapping[lab]
    return lab_out.astype(np.int32), int(mapping.max())


def compute_dice_and_iou(a, b):
    a = np.asarray(a).astype(bool, copy=False)
    b = np.asarray(b).astype(bool, copy=False)
    inter = int(np.count_nonzero(a & b))
    sa = int(np.count_nonzero(a))
    sb = int(np.count_nonzero(b))
    denom = sa + sb
    union = denom - inter
    dice = (2.0 * inter / denom) if denom > 0 else 1.0
    iou = (inter / union) if union > 0 else 1.0
    return float(dice), float(iou)


def _voxel_spacing(affine):
    return np.sqrt((np.asarray(affine)[:3, :3] ** 2).sum(axis=0))


class Evaluator(object):
    def __init__(self, cfg=None):
        cfg = cfg or {}
        ev = cfg.get('evaluation', {}) if 'evaluation' in cfg else cfg
        # A candidate counts as a detection hit (TP) if its segmented component
        # overlaps a GT lesion with IoU >= this. Default 0.0 = any overlap
        # counts (as in the source repo); IoU/Dice remain reported as quality
        # metrics. Raise for stricter matching.
        self.iou_threshold = ev.get('iou_threshold', 0.0)
        self.min_size = ev.get('min_size', 5)
        self.connectivity = ev.get('connectivity', 26)

    # ------------------------------------------------------------------ #
    def evaluate(self, pred_mask_path, gt_mask_path, detections=None):
        """Return (candidates_df, global_metrics_dict)."""
        pred_img = nib.load(str(pred_mask_path))
        gt_img = nib.load(str(gt_mask_path))
        pred = pred_img.get_fdata() > 0
        gt = gt_img.get_fdata() > 0
        spacing = _voxel_spacing(pred_img.affine)
        vox_mm3 = float(np.prod(spacing))

        pred_lab, n_pred = connected_components_3d(pred, self.connectivity, self.min_size)
        gt_lab, n_gt = connected_components_3d(gt, self.connectivity, self.min_size)

        gt_masks = [(gt_lab == i) for i in range(1, n_gt + 1)]

        # --- match each predicted component to its best GT component --------
        pred_best = {}   # pred_id -> (best_gt_id, iou, dice)
        for pi in range(1, n_pred + 1):
            pmask = (pred_lab == pi)
            best = (0, 0.0, 0.0)  # gt_id, iou, dice
            for gi, gmask in enumerate(gt_masks, start=1):
                dice, iou = compute_dice_and_iou(pmask, gmask)
                if iou > best[1]:
                    best = (gi, iou, dice)
            pred_best[pi] = best

        detections = detections or []

        # --- one row per candidate detection --------------------------------
        rows = []
        gt_hit = set()
        for idx, det in enumerate(detections):
            cx, cy, cz = self._detection_center(det)
            pred_id = int(pred_lab[cx, cy, cz]) if self._in_bounds(cx, cy, cz, pred_lab.shape) else 0
            if pred_id == 0:
                pred_id = self._nearest_component(cx, cy, cz, pred_lab, n_pred)
            if pred_id > 0:
                gt_id, iou, dice = pred_best[pred_id]
                vol_vox = int(np.count_nonzero(pred_lab == pred_id))
            else:
                gt_id, iou, dice, vol_vox = 0, 0.0, 0.0, 0
            is_tp = bool(gt_id > 0 and iou >= self.iou_threshold)
            if is_tp:
                gt_hit.add(gt_id)
            nodule = float(det.get('nodule_score', np.nan))
            lesion = float(det.get('lesion_score', np.nan))
            rows.append({
                'candidate': idx,
                'nodule_score': nodule,
                'lesion_score': lesion,
                'combined_score': nodule * lesion if np.isfinite(nodule) and np.isfinite(lesion) else np.nan,
                'n_merged': det.get('n_merged', 1),
                'cx': cx, 'cy': cy, 'cz': cz,
                'pred_component': pred_id,
                'matched_gt': gt_id if gt_id > 0 else None,
                'iou': iou,
                'dice': dice,
                'volume_vox': vol_vox,
                'volume_mm3': vol_vox * vox_mm3,
                'is_tp': is_tp,
            })
        candidates_df = pd.DataFrame(rows)

        # --- global metrics --------------------------------------------------
        n_cand = len(rows)
        tp = int(candidates_df['is_tp'].sum()) if n_cand else 0
        fp = n_cand - tp
        detected_gt = len(gt_hit)
        fn = n_gt - detected_gt
        sensitivity = detected_gt / n_gt if n_gt else float('nan')
        precision = tp / n_cand if n_cand else float('nan')
        f1 = (2 * precision * sensitivity / (precision + sensitivity)
              if precision and sensitivity and (precision + sensitivity) > 0 else 0.0)
        tp_rows = candidates_df[candidates_df['is_tp']] if n_cand else candidates_df
        vol_dice, vol_iou = compute_dice_and_iou(pred, gt)

        global_metrics = {
            'n_gt_lesions': n_gt,
            'n_pred_components': n_pred,
            'n_candidates': n_cand,
            'true_positives': tp,
            'false_positives': fp,
            'false_negatives': fn,
            'gt_lesions_detected': detected_gt,
            'sensitivity': sensitivity,
            'precision': precision,
            'f1': f1,
            'mean_iou_tp': float(tp_rows['iou'].mean()) if len(tp_rows) else float('nan'),
            'mean_dice_tp': float(tp_rows['dice'].mean()) if len(tp_rows) else float('nan'),
            'volumetric_dice': vol_dice,
            'volumetric_iou': vol_iou,
        }
        return candidates_df, global_metrics

    def evaluate_to_files(self, pred_mask_path, gt_mask_path, detections,
                          candidates_csv, global_csv):
        candidates_df, global_metrics = self.evaluate(
            pred_mask_path, gt_mask_path, detections)
        candidates_df.to_csv(candidates_csv, index=False)
        pd.DataFrame([global_metrics]).to_csv(global_csv, index=False)
        return candidates_df, global_metrics

    # ------------------------------------------------------------------ #
    @staticmethod
    def _detection_center(det):
        if 'x' in det and 'y' in det and 'z' in det:
            return int(det['x']), int(det['y']), int(det['z'])
        return (int(round((det['x_min'] + det['x_max']) / 2)),
                int(round((det['y_min'] + det['y_max']) / 2)),
                int(round((det['z_min'] + det['z_max']) / 2)))

    @staticmethod
    def _in_bounds(x, y, z, shape):
        return 0 <= x < shape[0] and 0 <= y < shape[1] and 0 <= z < shape[2]

    @staticmethod
    def _nearest_component(x, y, z, lab, n):
        """Nearest non-zero component centroid to (x,y,z); 0 if none within reach."""
        if n == 0:
            return 0
        best_id, best_d = 0, np.inf
        for i in range(1, n + 1):
            coords = np.argwhere(lab == i)
            if coords.size == 0:
                continue
            c = coords.mean(0)
            d = float(np.linalg.norm(c - np.array([x, y, z])))
            if d < best_d:
                best_id, best_d = i, d
        # only accept if reasonably close (within ~a box radius)
        return best_id if best_d <= 20 else 0
