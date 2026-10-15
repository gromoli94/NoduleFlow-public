"""Test the longitudinal pipeline with candidate propagation.

Covers:
  1. strategy donor logic (S0/S1/S2),
  2. end-to-end S1 run on T0/T1/T2 (register follow-ups -> T0, propagate
     candidates from previous times, segment each timepoint),
  3. propagation effect on the prompt masks,
  4. the register-if-not-present caching (reuse of registered niftis).

Usage:
    python test/test_longitudinal.py
"""
import os
import sys
import json
import logging
from pathlib import Path

import yaml
import pandas as pd
import nibabel as nib

REPO_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO_ROOT))

from src.longitudinal import LongitudinalPipeline
from src.modules.properties.tracking import NoduleTracker


def setup_logging(name, outputFolder):
    os.makedirs(outputFolder, exist_ok=True)
    for h in logging.root.handlers[:]:
        logging.root.removeHandler(h)
    logging.basicConfig(filename=os.path.join(outputFolder, f"{name}.log"),
                        filemode='w', level=logging.INFO,
                        format='%(asctime)s - %(levelname)s - %(message)s')


def load_cfg(strategy):
    with open(REPO_ROOT / "configs" / "config.yaml") as f:
        cfg = yaml.safe_load(f)
    cfg['longitudinal']['strategy'] = strategy
    cfg['longitudinal']['growth_report'] = True
    cfg['longitudinal']['time_days'] = {'T0': 0, 'T1': 90, 'T2': 180}  # -> doubling time
    cfg['pipeline']['max_prompts'] = 2      # keep LesionLocator fast
    cfg['weights_root'] = str(REPO_ROOT)
    wd = cfg.get('network_weights_dir', 'network_weights')
    if not os.path.isabs(wd):
        cfg['network_weights_dir'] = str(REPO_ROOT / wd)
    ck = cfg['lesionlocator'].get('checkpoint')
    if ck and not os.path.isabs(ck):
        cfg['lesionlocator']['checkpoint'] = str(REPO_ROOT / ck)
    return cfg


def nvox(path):
    return int((nib.load(str(path)).get_fdata() > 0).sum())


def test_donor_logic():
    print("=== Unit: strategy donor logic ===")
    for strat, expected in [("S0", [[0], [1], [2]]),
                            ("S1", [[0], [0, 1], [0, 1, 2]]),
                            ("S2", [[0, 1, 2], [0, 1, 2], [0, 1, 2]])]:
        p = LongitudinalPipeline.__new__(LongitudinalPipeline)
        p.strategy = strat
        got = [p._donors(j, 3) for j in range(3)]
        assert got == expected, f"{strat}: {got} != {expected}"
        print(f"  {strat}: {got}  OK")


def main():
    BASE = Path(__file__).resolve().parent
    outputFolder = BASE / "testResults"
    setup_logging("testLongitudinal", outputFolder)

    test_donor_logic()

    timepoints = [("T0", BASE / "example_T0.nii.gz"),
                  ("T1", BASE / "example_T1.nii.gz"),
                  ("T2", BASE / "example_T2.nii.gz")]
    for _, p in timepoints:
        assert p.exists(), f"missing {p}"

    gt_masks = {"T0": str(BASE / "example_T0_mask.nii.gz"),
                "T1": str(BASE / "example_T1_mask.nii.gz"),
                "T2": str(BASE / "example_T2_mask.nii.gz")}

    print("\n=== End-to-end: strategy S1 ===")
    cfg = load_cfg("S1")
    use_mask = cfg.get('registration', {}).get('register_with_mask', False)

    # Reuse cache: skip the expensive pipeline (detection/registration/segmentation)
    # when its results are already on disk. Set NODULEFLOW_FORCE=1 to force a rerun.
    results_json = outputFolder / "longitudinal_results.json"
    force = os.environ.get("NODULEFLOW_FORCE", "0") not in ("0", "", "false", "False")
    pipe, reused = None, False
    if results_json.exists() and not force:
        print(f"  Reusing cached pipeline results ({results_json.name}); "
              f"set NODULEFLOW_FORCE=1 to rerun.")
        with open(results_json) as f:
            results = json.load(f)
        reused = True
    else:
        pipe = LongitudinalPipeline(cfg, checkIfExists=True)
        results = pipe.Run(timepoints, outputFolder, gt_masks=gt_masks)
        with open(results_json, "w") as f:
            json.dump(results, f, indent=2,
                      default=lambda o: o.item() if hasattr(o, 'item') else str(o))

    assert {"T0", "T1", "T2"}.issubset(results)
    for label in ("T0", "T1", "T2"):
        r = results[label]
        assert Path(r['mask']).exists(), f"{label} segmentation missing: {r['mask']}"
        assert Path(r['prompt_mask']).exists(), f"{label} prompt missing"
        # evaluation ran against a GT in the output's space (fixed-frame) AND
        # in each timepoint's own space
        assert 'metrics' in r, f"{label}: no fixed-frame metrics produced"
        assert 'metrics_own' in r, f"{label}: no own-space metrics produced"
        assert Path(r['global_csv']).exists() and Path(r['candidates_csv']).exists()
        # per-nodule properties CSV alongside the segmentation
        assert 'properties_csv' in r and Path(r['properties_csv']).exists(), \
            f"{label}: missing properties CSV"
        m, mo = r['metrics'], r['metrics_own']
        print(f"  {label}: image={Path(r['image']).name}  "
              f"prompt_vox={nvox(r['prompt_mask'])}  seg_vox={nvox(r['mask'])}")
        print(f"      fixed-frame: sens={m['sensitivity']:.2f} prec={m['precision']:.2f} "
              f"dice={m['volumetric_dice']:.3f}  (gt={Path(r['gt_used']).name})")
        print(f"      own-space  : sens={mo['sensitivity']:.2f} prec={mo['precision']:.2f} "
              f"dice={mo['volumetric_dice']:.3f}")

    # T1/T2 GT should have been warped into the fixed (T0) frame for eval
    for t in ("T1", "T2"):
        assert results[t]['gt_used'].endswith("_gt_reg.nii.gz"), \
            f"{t} GT was not warped to the fixed frame: {results[t]['gt_used']}"
        # own-space output was warped back to the timepoint's own grid
        assert 'mask_own' in results[t] and Path(results[t]['mask_own']).exists(), \
            f"{t} own-space mask missing"
        own_shape = nib.load(results[t]['mask_own']).shape
        gt_shape = nib.load(str(BASE / f"example_{t}_mask.nii.gz")).shape
        assert own_shape == gt_shape, f"{t} own mask shape {own_shape} != GT {gt_shape}"
        # own-space properties CSV emitted for follow-ups too
        assert 'properties_csv_own' in results[t] and Path(results[t]['properties_csv_own']).exists(), \
            f"{t}: missing own-space properties CSV"
    assert results["T0"]['gt_used'].endswith("example_T0_mask.nii.gz"), \
        "T0 GT should be used as-is (already in the fixed frame)"
    # T0 is the fixed timepoint: own-space metrics == fixed-frame metrics
    assert results["T0"]['metrics_own']['volumetric_dice'] == \
        results["T0"]['metrics']['volumetric_dice']

    # registered follow-up niftis were produced (T1, T2 -> T0). Names may carry a
    # `_masked` tag when register_with_mask is on; read them from the results.
    for t in ("T1", "T2"):
        img = results[t]['image']
        assert Path(img).exists() and img.endswith("_reg.nii.gz"), f"{t} not registered: {img}"
    print(f"  registered niftis present: "
          f"{Path(results['T1']['image']).name}, {Path(results['T2']['image']).name}")

    # propagation effect (paths taken from results -> robust to the _masked tag):
    #   T0 (donors={T0}) prompt == its own BB;
    #   T2 (donors={T0,T1,T2}) prompt >= its own warped BB.
    assert nvox(results['T0']['prompt_mask']) == nvox(results['T0']['warped_bb']), \
        "T0 prompt should equal its own candidates (no donors)"
    t2_prompt = nvox(results['T2']['prompt_mask'])
    t2_own_warped = nvox(results['T2']['warped_bb'])
    assert t2_prompt >= t2_own_warped, "T2 propagated prompt must include its own candidates"
    print(f"  propagation OK: T0 prompt={nvox(results['T0']['prompt_mask'])} (own only); "
          f"T2 prompt={t2_prompt} >= T2 own-warped={t2_own_warped}")

    # caching: repeating the exact registration call (same mask arguments as the
    # pipeline used) must reuse the mask-aware cache, not recompute. Only when we
    # actually built the pipeline (skipped in reuse mode -- it needs the net).
    if not reused:
        res = pipe.registrator.RegisterOrLoad(
            str(timepoints[0][1]), str(timepoints[1][1]), outputFolder,
            movingMasks=[(str(outputFolder / "example_T1_BB.nii.gz"), "BB")],
            fixedMask=(gt_masks["T0"] if use_mask else None),
            movingMask=(gt_masks["T1"] if use_mask else None),
            checkIfExists=True)
        assert res['reused'] is True, "expected cached registration to be reused"
        print(f"  caching OK: cached registration reused ({Path(res['warped_image']).name}).")

    # mask-aware cache: with register_with_mask on, fields are stored under a
    # distinct _masked name; the whole-image variant must NOT exist (so toggling
    # register_with_mask can never silently reuse a stale field).
    if use_mask:
        assert '_masked' in Path(results['T1']['image']).name, "expected _masked cache naming"
        assert (outputFolder / "example_T1_masked_phi_AB.h5").exists(), "masked field missing"
        assert not (outputFolder / "example_T1_phi_AB.h5").exists(), \
            "whole-image field should not exist for a masked run"
        print("  mask-aware cache OK: masked fields stored separately (_masked).")

    # cross-timepoint growth report
    assert 'growth' in results, "no growth report produced"
    g = results['growth']
    assert Path(g['long_csv']).exists() and Path(g['summary_csv']).exists()
    sdf = pd.read_csv(g['summary_csv'])
    assert g['n_tracks'] >= 1 and len(sdf) >= 1, "expected >=1 nodule track"
    for col in ('volume_change_pct', 'diameter_change_mm', 'mass_change_mg', 'vdt_days'):
        assert col in sdf.columns, f"growth summary missing {col}"
    # a track spanning >1 timepoint should exist (the growing nodule)
    assert (sdf['n_timepoints'] > 1).any(), "no nodule tracked across >1 timepoint"
    top = sdf.sort_values('n_timepoints', ascending=False).iloc[0]
    print(f"\n=== Growth report: {g['n_tracks']} track(s) ===")
    print(f"  top track spans {int(top['n_timepoints'])} timepoints "
          f"({top['timepoints']}): "
          f"vol {top['volume_baseline_mm3']:.0f}->{top['volume_last_mm3']:.0f} mm^3 "
          f"({top['volume_change_pct']:+.0f}%), "
          f"diam {top['diameter_baseline_mm']:.1f}->{top['diameter_last_mm']:.1f} mm, "
          f"VDT={top['vdt_days']:.0f} d")

    # growth plots: (re)generate from the cached common-frame outputs, so
    # iterating on the plotting is cheap and never reruns the pipeline.
    tps = [(label, results[label]['mask'], results[label]['image'])
           for label in ("T0", "T1", "T2")]
    # reference (GT) masks in the common frame -> overlaid in green
    ref_masks = [results[label].get('gt_used') for label in ("T0", "T1", "T2")]
    lon = cfg['longitudinal']
    min_tp = lon.get('plot_min_timepoints', 2)
    plot_paths = NoduleTracker(cfg).plot_tracks(
        tps, outputFolder, min_timepoints=min_tp, ref_masks=ref_masks,
        masks=lon.get('plot_masks', ['pipeline', 'reference']),
        plot_3d=lon.get('plot_3d', True), plot_lung=lon.get('plot_lung', True),
        time_days=lon.get('time_days'))
    assert len(plot_paths) >= 1, "expected at least one growth plot"
    for p in plot_paths:
        assert Path(p).exists() and p.endswith(".png"), f"missing plot {p}"
    n_expected = int((sdf['n_timepoints'] >= min_tp).sum())
    assert len(plot_paths) == n_expected, \
        f"{len(plot_paths)} plots != {n_expected} tracks with >= {min_tp} timepoints"
    print(f"\n=== Growth plots: {len(plot_paths)} figure(s) (>= {min_tp} timepoints) ===")
    for p in plot_paths:
        print(f"  {Path(p).name}")

    print("\nLongitudinal test PASSED: propagation + growth report + plots work.")


if __name__ == "__main__":
    main()
