"""Test the LesionLocator baseline on an example CT scan.

Uses ``test/example_T0_mask.nii.gz`` as the annotation to derive a box prompt,
runs LesionLocator on ``test/example_T0.nii.gz``, and checks that a lesion
segmentation is produced.

Usage:
    python test/test_lesionlocator.py
"""
import os
import sys
import logging
from pathlib import Path

import yaml
import numpy as np
import nibabel as nib

# Make the repository root importable so ``src`` is a package.
REPO_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO_ROOT))

from src.baselines.lesionlocator import LesionLocator


def setup_logging(processName, outputFolder, mode='w'):
    os.makedirs(outputFolder, exist_ok=True)
    log_file = os.path.join(outputFolder, f"{processName}.log")
    for handler in logging.root.handlers[:]:
        logging.root.removeHandler(handler)
    logging.basicConfig(filename=log_file,
                        filemode=mode,
                        format='%(asctime)s - %(levelname)s - %(message)s',
                        level=logging.INFO)
    return log_file


def resolve_paths(cfg):
    """Resolve the repo-relative checkpoint path to absolute."""
    ll = cfg.get('lesionlocator', {})
    ckpt = ll.get('checkpoint')
    if ckpt and not os.path.isabs(ckpt):
        ll['checkpoint'] = str(REPO_ROOT / ckpt)
    return cfg


def main():
    BASE_DIR = Path(__file__).resolve().parent

    niftiToTest = [BASE_DIR / "example_T0.nii.gz"]
    annToTest = [BASE_DIR / "example_T0_mask.nii.gz"]
    assert niftiToTest[0].exists(), f"Missing example scan: {niftiToTest[0]}"
    assert annToTest[0].exists(), f"Missing example annotation: {annToTest[0]}"

    outputFolder = BASE_DIR / "testResults"
    outputFolder.mkdir(parents=True, exist_ok=True)

    configFile = REPO_ROOT / "configs" / "baseline" / "lesionlocator.yaml"
    with open(configFile, "r") as f:
        cfg = yaml.safe_load(f)
    cfg = resolve_paths(cfg)

    assert os.path.isdir(cfg['lesionlocator']['checkpoint']), \
        f"Missing checkpoint folder: {cfg['lesionlocator']['checkpoint']}"

    logFile = setup_logging("testLesionLocator", outputFolder)

    ll = LesionLocator(cfg, logFile, checkIfExists=False)

    # Segment lesions using a box prompt derived from the annotation mask.
    ll.SegmentLesions(niftis=niftiToTest, annFiles=annToTest,
                      outputFolder=outputFolder, modality="box")

    # Verify the merged output was produced and is a non-empty segmentation.
    stem = niftiToTest[0].name[:-7]  # strip ".nii.gz"
    outPath = outputFolder / f"{stem}_{ll.NoduleTag}.nii.gz"

    print("\n=== LesionLocator output ===")
    if not outPath.exists():
        raise SystemExit(f"LesionLocator test FAILED: output missing at {outPath}")

    seg = nib.load(str(outPath)).get_fdata()
    n_voxels = int((seg > 0).sum())
    print(f"  [OK] lesion segmentation: {outPath}")
    print(f"       segmented voxels: {n_voxels}")

    if n_voxels == 0:
        raise SystemExit("LesionLocator test FAILED: segmentation is empty.")
    print("\nLesionLocator test PASSED: lesion segmentation produced.")


if __name__ == "__main__":
    main()
