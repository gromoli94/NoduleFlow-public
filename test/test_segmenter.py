"""Test the Segmenter module.

Verifies the modular dispatch: selecting the LesionLocator backend by name,
running segmentation on test/example_T0.nii.gz through the uniform Segment()
interface, and that an unknown backend raises a clear error.

Usage:
    python test/test_segmenter.py
"""
import os
import sys
import copy
import logging
from pathlib import Path

import yaml
import nibabel as nib

REPO_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO_ROOT))

from src.modules.segmentation.segmenter import Segmenter


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
    with open(REPO_ROOT / "configs" / "modules" / "segmenter.yaml") as f:
        cfg = yaml.safe_load(f)
    # resolve backend checkpoint (repo-relative -> absolute)
    ll = cfg.get('lesionlocator', {})
    ckpt = ll.get('checkpoint')
    if ckpt and not os.path.isabs(ckpt):
        ll['checkpoint'] = str(REPO_ROOT / ckpt)
    return cfg


def main():
    BASE_DIR = Path(__file__).resolve().parent
    nifti = BASE_DIR / "example_T0.nii.gz"
    ann = BASE_DIR / "example_T0_mask.nii.gz"
    assert nifti.exists() and ann.exists()

    outputFolder = BASE_DIR / "testResults"
    setup_logging("testSegmenter", outputFolder)

    cfg = load_cfg()

    # --- 1. backend selection ------------------------------------------------
    print("Available segmenter backends:", Segmenter.available_backends())
    assert "lesionlocator" in Segmenter.available_backends()

    # --- 2. unknown backend raises a clear error -----------------------------
    bad_cfg = copy.deepcopy(cfg)
    bad_cfg['segmenter']['model'] = 'does_not_exist'
    try:
        Segmenter(bad_cfg)
        raise SystemExit("Segmenter test FAILED: unknown backend did not raise.")
    except ValueError as e:
        print(f"Unknown backend correctly rejected: {e}")

    # --- 3. run the real backend through the uniform interface ---------------
    seg = Segmenter(cfg, checkIfExists=False)
    assert seg.name == "lesionlocator"
    outPath = outputFolder / f"{nifti.name[:-7]}_{seg.backend.NoduleTag}.nii.gz"
    if outPath.exists():
        outPath.unlink()

    seg.Segment(niftis=[nifti], annFiles=[ann], outputFolder=outputFolder, modality="box")

    print("\n=== Segmenter output ===")
    if not outPath.exists():
        raise SystemExit(f"Segmenter test FAILED: output missing at {outPath}")
    n_voxels = int((nib.load(str(outPath)).get_fdata() > 0).sum())
    print(f"  [OK] backend={seg.name}: {outPath}")
    print(f"       segmented voxels: {n_voxels}")
    if n_voxels == 0:
        raise SystemExit("Segmenter test FAILED: segmentation is empty.")

    print("\nSegmenter test PASSED: modular dispatch + LesionLocator segmentation OK.")


if __name__ == "__main__":
    main()
