"""Test the Classification module (false-positive reduction) standalone.

Loads the ResNet50 classifier and scores a couple of candidate boxes on
test/example_T0.nii.gz, checking that it returns well-formed probabilities.

Boxes use the detector's native format [z0, x0, y0, z1, x1, y1]; here we
synthesize a box around the known nodule and a background box, so the test
runs without invoking the detector.

Usage:
    python test/test_classifier.py
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

from src.modules.classification.classifier import Classifier


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
    with open(REPO_ROOT / "configs" / "modules" / "classifier.yaml") as f:
        cfg = yaml.safe_load(f)
    wd = cfg.get('network_weights_dir', 'network_weights')
    if not os.path.isabs(wd):
        cfg['network_weights_dir'] = str(REPO_ROOT / wd)
    return cfg


def synth_box_around(cx, cy, cz, shape, half=6):
    """Build a detector-format box [z0,x0,y0,z1,x1,y1] around an array-space centre.

    Mirrors the x-flip used when converting detector boxes to array coords, so a
    box centred on array voxel (cx,cy,cz) maps back to that location.
    """
    x_flipped = shape[0] - 1 - cx  # detector x runs opposite to array x
    x0, x1 = x_flipped - half, x_flipped + half
    y0, y1 = cy - half, cy + half
    z0, z1 = cz - half, cz + half
    return [z0, x0, y0, z1, x1, y1]


def main():
    BASE_DIR = Path(__file__).resolve().parent
    nifti = BASE_DIR / "example_T0.nii.gz"
    mask = BASE_DIR / "example_T0_mask.nii.gz"
    assert nifti.exists() and mask.exists()

    setup_logging("testClassifier", BASE_DIR / "testResults")

    cfg = load_cfg()
    clf = Classifier(cfg)

    shape = nib.load(str(nifti)).shape
    # nodule centre (array space) from the reference mask
    cen = np.argwhere(nib.load(str(mask)).get_fdata() > 0).mean(0).round().astype(int)
    nodule_box = synth_box_around(cen[0], cen[1], cen[2], shape)
    # an arbitrary background box (corner of the volume)
    background_box = synth_box_around(20, 20, 20, shape)

    boxes_per_nifti = [[nodule_box, background_box]]
    scores = clf.Classify([nifti], boxes_per_nifti)

    print("\n=== Classifier output ===")
    assert len(scores) == 1, "expected one entry per scan"
    per_scan = scores[0]
    assert len(per_scan) == 2, f"expected 2 scores, got {len(per_scan)}"
    for name, s in zip(["nodule_box", "background_box"], per_scan):
        print(f"  {name:15s}: lesion probability = {s:.4f}")
        assert 0.0 <= s <= 1.0, f"score out of range: {s}"

    print("\nClassifier test PASSED: returned well-formed probabilities for all boxes.")


if __name__ == "__main__":
    main()
