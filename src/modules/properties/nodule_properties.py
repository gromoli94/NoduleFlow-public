"""Nodule properties module.

Given a lesion segmentation / mask (and, optionally, the CT it was drawn on),
compute per-nodule morphometric and densitometric properties:

  * volume (voxels and mm^3),
  * centroid (voxel and mm),
  * bounding box,
  * RECIST-style axial diameters: long, short and their average (mm),
  * mean HU and mass (mg)  -- when the CT is provided.

The diameter/centroid/mass formulas are transferred from the source repo
(`GetLesionDiameter`, `short_axis_from_slice_with_endpoints`, `GetLesionCenter`,
and the `mass = volume * (mean_HU/1000 + 1)` relation).
"""
import numpy as np
import pandas as pd
import nibabel as nib
from scipy.ndimage import label as cc_label
from scipy.ndimage import binary_erosion
from scipy.spatial import ConvexHull
from scipy.spatial.distance import pdist


def _voxel_spacing(affine):
    return np.sqrt((np.asarray(affine)[:3, :3] ** 2).sum(axis=0))


def connected_components_3d(mask, connectivity=26, min_size=5):
    structure = np.ones((3, 3, 3), dtype=np.uint8)
    if connectivity == 6:
        structure = np.zeros((3, 3, 3), dtype=np.uint8)
        structure[1, 1, :] = 1
        structure[1, :, 1] = 1
        structure[:, 1, 1] = 1
    lab, n = cc_label(mask.astype(bool), structure=structure)
    if n == 0:
        return lab.astype(np.int32), 0
    counts = np.bincount(lab.ravel())
    keep = np.zeros(n + 1, dtype=bool)
    keep[1:] = counts[1:] >= min_size
    mapping = np.zeros(n + 1, dtype=np.int32)
    mapping[keep] = np.cumsum(keep.astype(np.int32))[keep]
    return mapping[lab].astype(np.int32), int(mapping.max())


def _boundary_points(mask2d):
    """Boundary voxels of a 2D binary mask, as (axis0, axis1) integer coords."""
    mask2d = mask2d.astype(bool)
    if mask2d.sum() == 0:
        return np.zeros((0, 2), dtype=np.int32)
    er = binary_erosion(mask2d, structure=np.ones((3, 3), dtype=bool), border_value=0)
    return np.argwhere(mask2d & ~er).astype(np.int32)


def _max_pair_distance(points):
    """Maximum pairwise Euclidean distance among 2D points (mm), + the endpoints."""
    n = points.shape[0]
    if n < 2:
        return 0.0, (0, 0)
    if n <= 3:
        diff = points[:, None, :] - points[None, :, :]
        dist = np.sqrt((diff ** 2).sum(axis=2))
        ij = np.unravel_index(np.argmax(dist), dist.shape)
        return float(dist[ij]), (int(ij[0]), int(ij[1]))
    try:
        hull_idx = ConvexHull(points).vertices
    except Exception:
        hull_idx = np.arange(n)
    hull_pts = points[hull_idx]
    dists = pdist(hull_pts)
    max_k = int(np.argmax(dists))
    i_arr, j_arr = np.triu_indices(len(hull_pts), k=1)
    return float(dists[max_k]), (int(hull_idx[i_arr[max_k]]), int(hull_idx[j_arr[max_k]]))


def _short_axis(points_mm, u, nbins=25, min_pts=10):
    """Max width perpendicular to the long axis `u`, over binned long-axis slabs."""
    if points_mm.shape[0] < 2:
        return 0.0
    u = np.asarray(u, dtype=np.float64)
    un = float(np.linalg.norm(u))
    if un < 1e-8:
        return 0.0
    u = u / un
    nrm = np.array([-u[1], u[0]])
    t = points_mm @ u
    s = points_mm @ nrm
    t_min, t_max = float(t.min()), float(t.max())
    if t_max - t_min < 1e-6:
        return 0.0
    npts = points_mm.shape[0]
    nbins_eff = int(np.clip(npts // 30, 5, nbins))
    min_pts_eff = min(min_pts, max(2, npts // 20))
    edges = np.linspace(t_min, t_max, nbins_eff + 1)
    best = 0.0
    for k in range(nbins_eff):
        sel = ((t >= edges[k]) & (t <= edges[k + 1])) if k == nbins_eff - 1 \
            else ((t >= edges[k]) & (t < edges[k + 1]))
        if np.count_nonzero(sel) < min_pts_eff:
            continue
        s_sel = s[sel]
        best = max(best, float(s_sel.max() - s_sel.min()))
    if best == 0.0:
        best = float(s.max() - s.min())
    return best


def _lesion_diameter(comp3d, sp0, sp1):
    """RECIST-style diameters on the max-area axial slice (slices along axis 2).

    Returns (key_slice, long_mm, short_mm, avg_mm). `sp0`/`sp1` are the physical
    spacings (mm) of in-plane axes 0 and 1.
    """
    mask3d = comp3d.astype(bool)
    if mask3d.sum() == 0:
        return None, 0.0, 0.0, 0.0
    areas = mask3d.sum(axis=(0, 1))            # area per axis-2 slice
    z = int(np.argmax(areas))
    mask2d = mask3d[:, :, z]
    if mask2d.sum() == 0:
        return z, 0.0, 0.0, 0.0
    bnd = _boundary_points(mask2d)             # (axis0, axis1)
    if bnd.shape[0] == 0:
        return z, 0.0, 0.0, 0.0
    pts_mm = np.stack([bnd[:, 0] * sp0, bnd[:, 1] * sp1], axis=1)
    long_mm, (i, j) = _max_pair_distance(pts_mm)
    if long_mm <= 0:
        return z, 0.0, 0.0, 0.0
    short_mm = _short_axis(pts_mm, pts_mm[j] - pts_mm[i])
    return z, float(long_mm), float(short_mm), float(0.5 * (long_mm + short_mm))


class NoduleProperties(object):
    def __init__(self, cfg=None):
        cfg = cfg or {}
        p = cfg.get('properties', cfg)
        self.min_size = p.get('min_size', 5)
        self.connectivity = p.get('connectivity', 26)
        self.clip_HU = p.get('clip_HU', [-1000.0, 1000.0])

    def compute(self, mask_path, ct_path=None):
        """Return a list of per-nodule property dicts for the mask.

        If `ct_path` is given (same grid as the mask), mean HU and mass are added.
        """
        mimg = nib.load(str(mask_path))
        mask = mimg.get_fdata()
        spacing = _voxel_spacing(mimg.affine)          # (s0, s1, s2)
        vox_vol = float(np.prod(spacing))
        lab, n = connected_components_3d(mask > 0, self.connectivity, self.min_size)

        ct = None
        if ct_path is not None:
            ct = np.clip(nib.load(str(ct_path)).get_fdata(),
                         self.clip_HU[0], self.clip_HU[1])
            if ct.shape != mask.shape:
                raise ValueError(f"CT shape {ct.shape} != mask shape {mask.shape}")

        nodules = []
        for i in range(1, n + 1):
            comp = (lab == i)
            coords = np.argwhere(comp)                  # (N, 3) voxel (a0, a1, a2)
            vol_vox = int(coords.shape[0])
            cen = coords.mean(axis=0)                   # (a0, a1, a2)
            mn = coords.min(axis=0)
            mx = coords.max(axis=0)
            z_key, long_mm, short_mm, diam_mm = _lesion_diameter(comp, spacing[0], spacing[1])
            props = {
                'nodule_id': i,
                'volume_vox': vol_vox,
                'volume_mm3': vol_vox * vox_vol,
                'centroid_x': float(cen[0]), 'centroid_y': float(cen[1]), 'centroid_z': float(cen[2]),
                'centroid_x_mm': float(cen[0] * spacing[0]),
                'centroid_y_mm': float(cen[1] * spacing[1]),
                'centroid_z_mm': float(cen[2] * spacing[2]),
                'x_min': int(mn[0]), 'x_max': int(mx[0]),
                'y_min': int(mn[1]), 'y_max': int(mx[1]),
                'z_min': int(mn[2]), 'z_max': int(mx[2]),
                'key_slice': z_key,
                'diameter_long_mm': long_mm,
                'diameter_short_mm': short_mm,
                'diameter_mm': diam_mm,
            }
            if ct is not None:
                mean_hu = float(ct[comp].mean())
                props['mean_HU'] = mean_hu
                props['mass_mg'] = float(props['volume_mm3'] * (mean_hu / 1000.0 + 1.0))
            nodules.append(props)
        return nodules

    def to_dataframe(self, mask_path, ct_path=None):
        return pd.DataFrame(self.compute(mask_path, ct_path))

    def compute_to_csv(self, mask_path, out_csv, ct_path=None):
        df = self.to_dataframe(mask_path, ct_path)
        df.to_csv(out_csv, index=False)
        return df
