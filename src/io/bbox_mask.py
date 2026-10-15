"""Convert detector box outputs into a label-mask NIfTI.

The mask lives in the same voxel space as the reference CT (same shape/affine),
with each detection painted as a filled box. It is the bridge between the
Detector (which returns boxes) and prompt-driven segmenters such as
LesionLocator (which consume annotation masks / instance maps).
"""
import numpy as np
import nibabel as nib


def detections_to_bbox_mask(detections, ref_nifti_path, out_path, instance=False):
    """Paint detections as filled boxes into a mask aligned with the reference CT.

    Parameters
    ----------
    detections : list of dict
        Each with integer box corners x_min..z_max (voxel indices of the CT).
    ref_nifti_path : str or Path
        CT NIfTI that defines the output shape and affine.
    out_path : str or Path
        Where to write the mask.
    instance : bool
        If True, each box gets a distinct label (1..N); otherwise all boxes are
        labelled 1 (semantic / binary mask).

    Returns
    -------
    str -- out_path.
    """
    img = nib.load(str(ref_nifti_path))
    shape = img.shape
    dtype = np.uint16 if instance else np.uint8
    mask = np.zeros(shape, dtype=dtype)

    for i, det in enumerate(detections, start=1):
        x0 = int(np.clip(det["x_min"], 0, shape[0]))
        x1 = int(np.clip(det["x_max"], 0, shape[0]))
        y0 = int(np.clip(det["y_min"], 0, shape[1]))
        y1 = int(np.clip(det["y_max"], 0, shape[1]))
        z0 = int(np.clip(det["z_min"], 0, shape[2]))
        z1 = int(np.clip(det["z_max"], 0, shape[2]))
        if x1 <= x0 or y1 <= y0 or z1 <= z0:
            continue
        mask[x0:x1, y0:y1, z0:z1] = i if instance else 1

    header = img.header.copy()
    header.set_data_dtype(dtype)
    out_img = nib.Nifti1Image(mask, affine=img.affine, header=header)
    nib.save(out_img, str(out_path))
    return str(out_path)
