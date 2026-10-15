"""Test the Nodule Properties module.

Computes per-nodule properties from a lesion mask (with and without the CT),
and checks a synthetic two-sphere mask with known geometry.

Usage:
    python test/test_properties.py
"""
import tempfile
import os
import sys
from pathlib import Path

import yaml
import numpy as np
import nibabel as nib

REPO_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO_ROOT))

from src.modules.properties.nodule_properties import NoduleProperties


def load_cfg():
    with open(REPO_ROOT / "configs" / "modules" / "properties.yaml") as f:
        return yaml.safe_load(f)


def test_real(np_):
    BASE = Path(__file__).resolve().parent
    mask = BASE / "example_T2_mask.nii.gz"
    ct = BASE / "example_T2.nii.gz"
    assert mask.exists() and ct.exists()

    prof = NoduleProperties(load_cfg())

    # with CT -> HU + mass present
    nods = prof.compute(mask, ct_path=ct)
    print(f"=== Real mask: {len(nods)} nodule(s) ===")
    assert len(nods) >= 1
    for d in nods:
        print(f"  #{d['nodule_id']}: vol={d['volume_mm3']:.1f} mm^3  "
              f"diam(long/short/avg)={d['diameter_long_mm']:.1f}/"
              f"{d['diameter_short_mm']:.1f}/{d['diameter_mm']:.1f} mm  "
              f"HU={d['mean_HU']:.1f}  mass={d['mass_mg']:.1f} mg  "
              f"centroid=({d['centroid_x']:.0f},{d['centroid_y']:.0f},{d['centroid_z']:.0f})")
        assert d['volume_mm3'] > 0
        assert d['diameter_long_mm'] >= d['diameter_short_mm'] > 0
        assert d['diameter_long_mm'] >= d['diameter_mm'] >= d['diameter_short_mm']
        assert -1000.0 <= d['mean_HU'] <= 1000.0
        assert 'mass_mg' in d

    # without CT -> no HU/mass keys
    nods_no_ct = prof.compute(mask)
    assert all('mean_HU' not in d and 'mass_mg' not in d for d in nods_no_ct)
    assert len(nods_no_ct) == len(nods)
    print("  without CT: HU/mass omitted, same nodule count. OK")


def test_synthetic():
    """Two solid spheres on a 1 mm isotropic grid; check count, volume, diameter."""
    print("\n=== Synthetic: two spheres ===")
    shape = (80, 80, 40)
    vol = np.zeros(shape, dtype=np.uint8)
    zz, yy, xx = np.ogrid[:shape[0], :shape[1], :shape[2]]

    def sphere(cx, cy, cz, r):
        return ((zz - cx) ** 2 + (yy - cy) ** 2 + (xx - cz) ** 2) <= r ** 2

    r1, r2 = 6, 4
    vol[sphere(20, 20, 20, r1)] = 1
    vol[sphere(60, 60, 20, r2)] = 1
    affine = np.diag([1.0, 1.0, 1.0, 1.0])  # 1 mm isotropic

    tmp = Path(tempfile.mkdtemp())
    tmp.mkdir(parents=True, exist_ok=True)
    mpath = tmp / "two_spheres.nii.gz"
    nib.save(nib.Nifti1Image(vol, affine), str(mpath))

    prof = NoduleProperties({'min_size': 5, 'connectivity': 26})
    nods = prof.compute(mpath)
    assert len(nods) == 2, f"expected 2 nodules, got {len(nods)}"
    nods = sorted(nods, key=lambda d: d['volume_mm3'], reverse=True)
    for d, r in zip(nods, [r1, r2]):
        exp_vol = 4.0 / 3.0 * np.pi * r ** 3
        exp_diam = 2 * r
        print(f"  sphere r={r}: vol={d['volume_mm3']:.0f} (~{exp_vol:.0f}) mm^3  "
              f"long_diam={d['diameter_long_mm']:.1f} (~{exp_diam}) mm")
        # discretized sphere volume within ~15% of the analytic value
        assert abs(d['volume_mm3'] - exp_vol) / exp_vol < 0.15
        # long axis within ~1.5 voxels of the true diameter
        assert abs(d['diameter_long_mm'] - exp_diam) <= 1.5
    print("  synthetic geometry matches expectations. OK")


def main():
    test_real(np)
    test_synthetic()
    print("\nProperties test PASSED: per-nodule diameters, volume, HU and mass computed.")


if __name__ == "__main__":
    main()
