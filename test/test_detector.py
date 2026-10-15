"""Test the Detector module on an example CT scan.

Exercises the pipeline choices exposed in configs/modules/detector.yaml:
  - detection model      (DLCSD-mD / LUNA16-mD)
  - classifier module    (enabled / disabled)
  - output coordinates   (box / point)

Runs NoduleNet-style detection on test/example_T0.nii.gz and checks that each
configuration produces well-formed detections.

Usage:
    python test/test_detector.py
"""
import os
import sys
import copy
import logging
from pathlib import Path

import yaml

# Make the repository root importable so ``src`` is a package.
REPO_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO_ROOT))

from src.modules.detection.detector import Detector


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


def load_cfg():
    configFile = REPO_ROOT / "configs" / "modules" / "detector.yaml"
    with open(configFile, "r") as f:
        cfg = yaml.safe_load(f)
    # resolve weights dir to absolute
    wd = cfg.get('network_weights_dir', 'network_weights')
    if not os.path.isabs(wd):
        cfg['network_weights_dir'] = str(REPO_ROOT / wd)
    return cfg


BOX_KEYS = {"x_min", "x_max", "y_min", "y_max", "z_min", "z_max"}
POINT_KEYS = {"x", "y", "z"}


def check_detections(dets, output_mode):
    """Validate detection structure; return the count."""
    assert isinstance(dets, list)
    for d in dets:
        assert "nodule_score" in d and "lesion_score" in d
        if output_mode == "box":
            assert BOX_KEYS.issubset(d), f"missing box keys: {d}"
            assert d["x_max"] >= d["x_min"] and d["y_max"] >= d["y_min"] and d["z_max"] >= d["z_min"]
        else:
            assert POINT_KEYS.issubset(d), f"missing point keys: {d}"
    return len(dets)


def run_scenario(name, overrides, nifti):
    cfg = load_cfg()
    det_cfg = cfg['detector']
    det_cfg.update(overrides.get('detector', {}))
    if 'classifier' in overrides:
        det_cfg['classifier'].update(overrides['classifier'])
    print(f"\n{'='*70}\nScenario: {name}\n{'='*70}")
    detector = Detector(cfg)
    results = detector.Detect([nifti])
    dets = results[str(nifti)]
    n = check_detections(dets, det_cfg['output'])
    print(f"  -> {n} detection(s), output={det_cfg['output']}, "
          f"classifier={'on' if det_cfg['classifier']['enabled'] else 'off'}, "
          f"model={det_cfg['model']}")
    if dets:
        print(f"  sample detection: {dets[0]}")
    return n


def main():
    BASE_DIR = Path(__file__).resolve().parent
    nifti = BASE_DIR / "example_T0.nii.gz"
    assert nifti.exists(), f"Missing example scan: {nifti}"

    outputFolder = BASE_DIR / "testResults"
    setup_logging("testDetector", outputFolder)

    scenarios = [
        ("DLCSD-mD + classifier + box",
         {"detector": {"model": "DLCSD-mD", "output": "box"},
          "classifier": {"enabled": True}}),
        ("DLCSD-mD, no classifier, point",
         {"detector": {"model": "DLCSD-mD", "output": "point"},
          "classifier": {"enabled": False}}),
        ("LUNA16-mD + classifier + box",
         {"detector": {"model": "LUNA16-mD", "output": "box"},
          "classifier": {"enabled": True}}),
    ]

    counts = {}
    for name, overrides in scenarios:
        counts[name] = run_scenario(name, overrides, nifti)

    print(f"\n{'='*70}\n==================== Detector Test Summary ====================")
    for name, n in counts.items():
        print(f"  {name:38s}: {n} detection(s)")
    print('='*70)

    # At least one configuration should detect something on a scan with a nodule.
    if sum(counts.values()) == 0:
        raise SystemExit("Detector test FAILED: no detections in any configuration.")
    print("\nDetector test PASSED: all pipeline configurations ran and returned well-formed detections.")


if __name__ == "__main__":
    main()
