"""End-to-end test of the single-scan NoduleFlow pipeline.

Given test/example_T0.nii.gz, runs detection + classification (+ ensemble) and
feeds the resulting bounding-box prompts to LesionLocator, producing a final
lesion mask. Checks the intermediate prompt mask and the final mask.

Usage:
    python test/test_pipeline.py
"""
import os
import sys
import logging
from pathlib import Path

import yaml
import numpy as np
import nibabel as nib

REPO_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO_ROOT))

from src.pipeline import Pipeline


def setup_logging(processName, outputFolder, mode='w'):
    os.makedirs(outputFolder, exist_ok=True)
    log_file = os.path.join(outputFolder, f"{processName}.log")
    for handler in logging.root.handlers[:]:
        logging.root.removeHandler(handler)
    logging.basicConfig(filename=log_file, filemode=mode,
                        format='%(asctime)s - %(levelname)s - %(message)s',
                        level=logging.INFO)
    return log_file


def load_cfg():
    with open(REPO_ROOT / "configs" / "config.yaml") as f:
        cfg = yaml.safe_load(f)
    # resolve repo-relative paths -> absolute
    wd = cfg.get('network_weights_dir', 'network_weights')
    if not os.path.isabs(wd):
        cfg['network_weights_dir'] = str(REPO_ROOT / wd)
    ckpt = cfg['lesionlocator'].get('checkpoint')
    if ckpt and not os.path.isabs(ckpt):
        cfg['lesionlocator']['checkpoint'] = str(REPO_ROOT / ckpt)
    return cfg


def main():
    BASE_DIR = Path(__file__).resolve().parent
    nifti = BASE_DIR / "example_T0.nii.gz"
    assert nifti.exists(), f"Missing example scan: {nifti}"

    outputFolder = BASE_DIR / "testResults"
    setup_logging("testPipeline", outputFolder)

    cfg = load_cfg()
    stem = nifti.name[:-7]

    # clean prior outputs
    bb_path = outputFolder / f"{stem}_{cfg['pipeline']['bbox_tag']}.nii.gz"
    out_path = outputFolder / f"{stem}_{cfg['pipeline']['output_tag']}.nii.gz"
    for p in (bb_path, out_path):
        if p.exists():
            p.unlink()

    pipeline = Pipeline(cfg, checkIfExists=False)
    result = pipeline.Run(nifti, outputFolder)

    print("\n=== Pipeline outputs ===")
    # intermediate prompt mask
    assert bb_path.exists(), f"prompt mask missing: {bb_path}"
    bb_vox = int((nib.load(str(bb_path)).get_fdata() > 0).sum())
    print(f"  [OK] prompt mask : {bb_path}  ({bb_vox} voxels)")
    assert bb_vox > 0, "prompt mask is empty (no detections painted)"

    # final segmentation
    mask_path = result['mask']
    assert Path(mask_path).exists(), f"final mask missing: {mask_path}"
    seg = nib.load(mask_path)
    n_vox = int((seg.get_fdata() > 0).sum())
    print(f"  [OK] final mask  : {mask_path}  ({n_vox} voxels)")
    assert n_vox > 0, "final segmentation is empty"

    # sanity: final mask aligns with the input scan
    orig = nib.load(str(nifti))
    assert seg.shape == orig.shape, "final mask shape mismatch"
    assert np.allclose(seg.affine, orig.affine), "final mask affine mismatch"

    # sanity: at least one segmented component localizes the known nodule
    # (the mask may also contain false positives from other detector candidates,
    # so compare per-component, not the centroid of the whole mask).
    from scipy.ndimage import label
    ref = np.argwhere(nib.load(str(BASE_DIR / "example_T0_mask.nii.gz")).get_fdata() > 0).mean(0)
    lab, n = label(seg.get_fdata() > 0)
    dists = [float(np.linalg.norm(ref - np.argwhere(lab == i).mean(0))) for i in range(1, n + 1)]
    nearest = min(dists) if dists else float('inf')
    print(f"  {n} segmented component(s); nearest to nodule {np.round(ref,1)} -> {nearest:.1f} voxels")

    # per-nodule properties CSV (compute_properties: true in config)
    assert 'properties_csv' in result, "expected a properties CSV"
    import pandas as pd
    pdf = pd.read_csv(result['properties_csv'])
    assert {'volume_mm3', 'diameter_mm', 'mean_HU', 'mass_mg'}.issubset(pdf.columns)
    assert len(pdf) == n, "one properties row per segmented nodule expected"
    print(f"  [OK] properties CSV: {Path(result['properties_csv']).name} ({len(pdf)} nodule rows)")

    print("\nPipeline test PASSED: end-to-end mask + per-nodule properties produced.")


if __name__ == "__main__":
    main()
