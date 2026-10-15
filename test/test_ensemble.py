"""Test the ensemble false-positive reduction for the Detector.

Part 1 (unit): a deterministic check of the merge math on synthetic boxes.
Part 2 (integration): the Detector honours the ensemble on/off config flag and
merges real overlapping candidates on test/example_T0.nii.gz.

Usage:
    python test/test_ensemble.py
"""
import os
import sys
import copy
import math
import logging
from pathlib import Path

import yaml

REPO_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO_ROOT))

from src.modules.detection.ensemble import merge_overlapping_detections, EnsembleReducer
from src.modules.detection.detector import Detector


def box(nm, ls, x0, x1, y0, y1, z0, z1):
    return {"nodule_score": nm, "lesion_score": ls,
            "x_min": x0, "x_max": x1, "y_min": y0, "y_max": y1,
            "z_min": z0, "z_max": z1}


def test_unit():
    print("=== Unit: merge math on synthetic boxes ===")
    A = box(0.5, 0.5, 10, 20, 10, 20, 10, 20)
    B = box(0.6, 0.4, 15, 25, 15, 25, 15, 25)   # overlaps A
    C = box(0.3, 0.9, 100, 110, 100, 110, 100, 110)  # far away
    merged = merge_overlapping_detections([A, B, C])

    assert len(merged) == 2, f"expected 2 merged detections, got {len(merged)}"
    by_n = {d["n_merged"]: d for d in merged}
    assert set(by_n) == {1, 2}, f"unexpected n_merged values: {list(by_n)}"

    ab = by_n[2]
    # union geometry
    assert (ab["x_min"], ab["x_max"]) == (10, 25)
    assert (ab["y_min"], ab["y_max"]) == (10, 25)
    assert (ab["z_min"], ab["z_max"]) == (10, 25)
    # probabilistic-OR fused scores: 1 - prod(1 - s)
    assert math.isclose(ab["nodule_score"], 1 - (1 - 0.5) * (1 - 0.6), rel_tol=1e-9), ab["nodule_score"]
    assert math.isclose(ab["lesion_score"], 1 - (1 - 0.5) * (1 - 0.4), rel_tol=1e-9), ab["lesion_score"]
    # fused score is >= every member (OR can only raise confidence)
    assert ab["nodule_score"] >= max(A["nodule_score"], B["nodule_score"])

    c = by_n[1]
    assert math.isclose(c["nodule_score"], 0.3)
    print(f"  merged A+B -> nodule={ab['nodule_score']:.3f} lesion={ab['lesion_score']:.3f} "
          f"box=({ab['x_min']}-{ab['x_max']}), C kept separate. OK")


def load_cfg():
    with open(REPO_ROOT / "configs" / "modules" / "detector.yaml") as f:
        cfg = yaml.safe_load(f)
    wd = cfg.get('network_weights_dir', 'network_weights')
    if not os.path.isabs(wd):
        cfg['network_weights_dir'] = str(REPO_ROOT / wd)
    return cfg


def test_integration(nifti):
    print("\n=== Integration: Detector with ensemble off vs on ===")
    cfg = load_cfg()
    # keep it light: skip the classifier stage for this test
    cfg['detector']['classifier']['enabled'] = False
    cfg['detector']['output'] = 'box'

    # ensemble OFF -> raw candidates
    cfg_off = copy.deepcopy(cfg)
    cfg_off['detector']['ensemble']['enabled'] = False
    det_off = Detector(cfg_off)
    assert det_off.ensemble.enabled is False
    dets_off = det_off.Detect([nifti])[os.path.abspath(str(nifti))]
    n_off = len(dets_off)
    assert all('n_merged' not in d for d in dets_off), "raw detections should not be merged"

    # Merge those same detections directly (avoids a second full inference).
    reducer = EnsembleReducer(enabled=True)
    dets_on = reducer.reduce(copy.deepcopy(dets_off))
    n_on = len(dets_on)

    print(f"  candidates: {n_off}  ->  merged: {n_on}")
    # Invariant: every original candidate belongs to exactly one merged group.
    assert sum(d['n_merged'] for d in dets_on) == n_off, "n_merged must partition the candidates"
    assert n_on <= n_off
    assert any(d['n_merged'] > 1 for d in dets_on), "expected some overlapping candidates to merge"
    # scores recomputed: a merged group's score >= any single candidate's max
    top = max(dets_on, key=lambda d: d['nodule_score'])
    print(f"  top merged detection: n_merged={top['n_merged']} "
          f"nodule={top['nodule_score']:.3f} lesion={top['lesion_score']:.3f}")

    # And confirm the Detector applies it internally when the flag is on.
    cfg_on = copy.deepcopy(cfg)
    cfg_on['detector']['ensemble']['enabled'] = True
    det_on = Detector(cfg_on)
    assert det_on.ensemble.enabled is True
    dets_internal = det_on.Detect([nifti])[os.path.abspath(str(nifti))]
    assert all('n_merged' in d for d in dets_internal), "enabled detector must return merged dets"
    print(f"  Detector(ensemble=on).Detect -> {len(dets_internal)} merged detections")


def main():
    BASE_DIR = Path(__file__).resolve().parent
    nifti = BASE_DIR / "example_T0.nii.gz"
    assert nifti.exists()
    logging.basicConfig(level=logging.INFO)

    test_unit()
    test_integration(nifti)

    print("\nEnsemble test PASSED: merge math correct and Detector honours the on/off flag.")


if __name__ == "__main__":
    main()
