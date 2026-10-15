"""Test the Registration module (uniGradICON / GradICON).

Registers the follow-up timepoints (T1, T2) onto the baseline (T0), warping
each moving CT -- and its lesion mask -- onto the T0 grid. Checks that the
warped outputs are produced, sit on the fixed grid, and that image similarity
to T0 improves after registration.

Usage:
    python test/test_registration.py
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

from src.modules.registration.registrator import Registrator


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
    with open(REPO_ROOT / "configs" / "modules" / "registration.yaml") as f:
        cfg = yaml.safe_load(f)
    cfg['weights_root'] = str(REPO_ROOT)  # where network_weights/ lives
    return cfg


def main():
    BASE_DIR = Path(__file__).resolve().parent
    fixed = BASE_DIR / "example_T0.nii.gz"
    movings = [BASE_DIR / "example_T1.nii.gz", BASE_DIR / "example_T2.nii.gz"]
    moving_masks = [[(BASE_DIR / "example_T1_mask.nii.gz", "mask")],
                    [(BASE_DIR / "example_T2_mask.nii.gz", "mask")]]
    for p in [fixed] + movings:
        assert p.exists(), f"missing {p}"

    outputFolder = BASE_DIR / "testResults"
    setup_logging("testRegistration", outputFolder)

    cfg = load_cfg()
    reg = Registrator(cfg)

    results = reg.RegisterSeries(fixed, movings, outputFolder,
                                 movingMasksPerScan=moving_masks)

    fixed_img = nib.load(str(fixed))
    print("\n=== Registration results ===")
    assert len(results) == len(movings)
    for moving, res in zip(movings, results):
        wimg = res['warped_image']
        assert Path(wimg).exists(), f"warped image missing: {wimg}"
        w = nib.load(wimg)
        # warped image sits on the fixed (T0) grid
        assert w.shape == fixed_img.shape, f"shape {w.shape} != fixed {fixed_img.shape}"
        assert np.allclose(w.affine, fixed_img.affine, atol=1e-3), "warped affine != fixed affine"

        # warped mask exists, is on the fixed grid, and is non-empty
        wmask = res['warped_masks']['mask']
        assert Path(wmask).exists(), f"warped mask missing: {wmask}"
        wm = nib.load(wmask)
        assert wm.shape == fixed_img.shape
        n_vox = int((wm.get_fdata() > 0).sum())

        mb, ma = res['metrics_before'], res['metrics_after']
        print(f"  {moving.name}:")
        print(f"    warped image -> {wimg}")
        print(f"    warped mask  -> {wmask}  ({n_vox} nodule voxels)")
        print(f"    SSIM {mb['ssim']:.4f} -> {ma['ssim']:.4f} | "
              f"NCC {mb['ncc']:.4f} -> {ma['ncc']:.4f} | "
              f"MSE {mb['mse']:.1f} -> {ma['mse']:.1f}")

        assert n_vox > 0, "warped nodule mask is empty"
        # registration should not worsen alignment (allow tiny numerical slack)
        assert ma['ssim'] >= mb['ssim'] - 1e-3, \
            f"SSIM worsened after registration: {mb['ssim']} -> {ma['ssim']}"

    print("\nRegistration test PASSED: follow-up scans registered onto T0 with warped masks.")


if __name__ == "__main__":
    main()
