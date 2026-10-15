"""Test the evaluation part of the pipeline.

Runs the full pipeline on test/example_T0.nii.gz WITH the ground-truth mask
(example_T0_mask.nii.gz) and checks that, alongside the output mask, it writes:
  - a per-candidate CSV (scores + metrics per detection),
  - a global-metrics CSV.

Usage:
    python test/test_evaluation.py
"""
import os
import sys
import logging
from pathlib import Path

import yaml
import pandas as pd

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
    wd = cfg.get('network_weights_dir', 'network_weights')
    if not os.path.isabs(wd):
        cfg['network_weights_dir'] = str(REPO_ROOT / wd)
    ckpt = cfg['lesionlocator'].get('checkpoint')
    if ckpt and not os.path.isabs(ckpt):
        cfg['lesionlocator']['checkpoint'] = str(REPO_ROOT / ckpt)
    return cfg


CANDIDATE_COLS = {'candidate', 'nodule_score', 'lesion_score', 'combined_score',
                  'pred_component', 'matched_gt', 'iou', 'dice', 'volume_mm3', 'is_tp'}
GLOBAL_COLS = {'n_gt_lesions', 'n_candidates', 'true_positives', 'false_positives',
               'false_negatives', 'sensitivity', 'precision', 'f1',
               'volumetric_dice', 'volumetric_iou'}


def main():
    BASE_DIR = Path(__file__).resolve().parent
    nifti = BASE_DIR / "example_T0.nii.gz"
    gt = BASE_DIR / "example_T0_mask.nii.gz"
    assert nifti.exists() and gt.exists()

    outputFolder = BASE_DIR / "testResults"
    setup_logging("testEvaluation", outputFolder)

    cfg = load_cfg()
    stem = nifti.name[:-7]
    cand_csv = outputFolder / f"{stem}_candidates.csv"
    glob_csv = outputFolder / f"{stem}_global.csv"
    for p in (cand_csv, glob_csv):
        if p.exists():
            p.unlink()

    pipeline = Pipeline(cfg, checkIfExists=False)
    result = pipeline.Run(nifti, outputFolder, gt_mask=str(gt))

    print("\n=== Evaluation outputs ===")
    for key in ('mask', 'candidates_csv', 'global_csv', 'metrics'):
        assert key in result, f"missing result key: {key}"

    # per-candidate file
    assert cand_csv.exists(), f"candidates CSV missing: {cand_csv}"
    cdf = pd.read_csv(cand_csv)
    print(f"  [OK] candidates CSV: {cand_csv}  ({len(cdf)} rows)")
    assert CANDIDATE_COLS.issubset(cdf.columns), \
        f"missing candidate columns: {CANDIDATE_COLS - set(cdf.columns)}"
    assert len(cdf) == result['metrics']['n_candidates']
    # at least one true positive (the pipeline captures the known nodule)
    assert cdf['is_tp'].any(), "no candidate matched the ground-truth nodule"
    print(cdf[['candidate', 'nodule_score', 'lesion_score', 'iou', 'dice', 'is_tp']].to_string(index=False))

    # global file
    assert glob_csv.exists(), f"global CSV missing: {glob_csv}"
    gdf = pd.read_csv(glob_csv)
    print(f"\n  [OK] global CSV: {glob_csv}")
    assert GLOBAL_COLS.issubset(gdf.columns), \
        f"missing global columns: {GLOBAL_COLS - set(gdf.columns)}"
    assert len(gdf) == 1
    m = result['metrics']
    print(f"  n_gt={m['n_gt_lesions']} TP={m['true_positives']} FP={m['false_positives']} "
          f"FN={m['false_negatives']}")
    print(f"  sensitivity={m['sensitivity']:.3f} precision={m['precision']:.3f} "
          f"f1={m['f1']:.3f} vol_dice={m['volumetric_dice']:.3f}")

    assert m['n_gt_lesions'] >= 1
    assert m['true_positives'] >= 1, "expected at least one detected GT lesion"
    assert 0.0 <= m['sensitivity'] <= 1.0

    print("\nEvaluation test PASSED: per-candidate and global metric files produced.")


if __name__ == "__main__":
    main()
