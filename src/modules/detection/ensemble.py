"""Ensemble false-positive reduction for the Detector.

Spatially overlapping candidate boxes are grouped into connected components
(union-find over pairwise box intersection) and each group is merged into a
single detection:

  * geometry  -> the union (axis-aligned bounding box) of the members,
  * scores    -> fused with a probabilistic OR:  1 - prod(1 - s),
  * n_merged  -> how many candidates were merged.

Adapted from `BBoxToMaskConverter.unify_intersecting_boxes` in the source repo;
the ground-truth `lesion_id` gate is dropped because it is not available at
inference time -- here candidates merge on spatial overlap alone.
"""
import numpy as np

_BOX_KEYS = ("x_min", "x_max", "y_min", "y_max", "z_min", "z_max")


def _prob_or(scores):
    """Probabilistic OR of independent detection scores: 1 - prod(1 - s)."""
    scores = np.asarray(scores, dtype=float)
    scores = scores[np.isfinite(scores)]
    if scores.size == 0:
        return np.nan
    scores = np.clip(scores, 0.0, 1.0)
    return float(1.0 - np.prod(1.0 - scores))


def merge_overlapping_detections(detections, touch_counts_as_intersection=True):
    """Merge spatially overlapping box detections into single detections.

    Parameters
    ----------
    detections : list of dict
        Each must have integer box corners (x_min..z_max) plus 'nodule_score'
        and 'lesion_score'.
    touch_counts_as_intersection : bool
        If True, boxes that merely touch (shared face/edge) are merged (<=);
        if False, they must strictly overlap (<).

    Returns
    -------
    list of dict -- merged detections with union geometry, prob-OR fused scores
    and an added 'n_merged' count. Order is by descending nodule_score.
    """
    n = len(detections)
    if n <= 1:
        for d in detections:
            d.setdefault("n_merged", 1)
        return detections

    xmin = np.array([d["x_min"] for d in detections], float)
    xmax = np.array([d["x_max"] for d in detections], float)
    ymin = np.array([d["y_min"] for d in detections], float)
    ymax = np.array([d["y_max"] for d in detections], float)
    zmin = np.array([d["z_min"] for d in detections], float)
    zmax = np.array([d["z_max"] for d in detections], float)

    parent = list(range(n))

    def find(a):
        while parent[a] != a:
            parent[a] = parent[parent[a]]
            a = parent[a]
        return a

    def union(a, b):
        ra, rb = find(a), find(b)
        if ra != rb:
            parent[rb] = ra

    def intersects(i, j):
        if touch_counts_as_intersection:
            ix = max(xmin[i], xmin[j]) <= min(xmax[i], xmax[j])
            iy = max(ymin[i], ymin[j]) <= min(ymax[i], ymax[j])
            iz = max(zmin[i], zmin[j]) <= min(zmax[i], zmax[j])
        else:
            ix = max(xmin[i], xmin[j]) < min(xmax[i], xmax[j])
            iy = max(ymin[i], ymin[j]) < min(ymax[i], ymax[j])
            iz = max(zmin[i], zmin[j]) < min(zmax[i], zmax[j])
        return ix and iy and iz

    # Connected components of intersecting boxes (O(n^2) per scan).
    for i in range(n):
        for j in range(i + 1, n):
            if intersects(i, j):
                union(i, j)

    # Group member indices by component root.
    comps = {}
    for i in range(n):
        comps.setdefault(find(i), []).append(i)

    merged = []
    for members in comps.values():
        group = [detections[i] for i in members]
        fused_nodule = _prob_or([d["nodule_score"] for d in group])
        fused_lesion = _prob_or([d["lesion_score"] for d in group])
        det = {
            "nodule_score": fused_nodule,
            "lesion_score": fused_lesion,
            "x_min": int(min(d["x_min"] for d in group)),
            "x_max": int(max(d["x_max"] for d in group)),
            "y_min": int(min(d["y_min"] for d in group)),
            "y_max": int(max(d["y_max"] for d in group)),
            "z_min": int(min(d["z_min"] for d in group)),
            "z_max": int(max(d["z_max"] for d in group)),
            "n_merged": len(group),
        }
        merged.append(det)

    merged.sort(key=lambda d: d["nodule_score"], reverse=True)
    return merged


class EnsembleReducer(object):
    """Config-driven wrapper around `merge_overlapping_detections`."""

    def __init__(self, cfg=None, enabled=False, touch_counts_as_intersection=True):
        if cfg is not None:
            enabled = cfg.get("enabled", enabled)
            touch_counts_as_intersection = cfg.get(
                "touch_counts_as_intersection", touch_counts_as_intersection)
        self.enabled = enabled
        self.touch_counts_as_intersection = touch_counts_as_intersection

    def reduce(self, detections):
        """Merge overlapping detections; assumes the reducer is enabled."""
        return merge_overlapping_detections(
            detections, touch_counts_as_intersection=self.touch_counts_as_intersection)
