"""Cross-timepoint nodule tracking, growth report, and growth plots.

Given per-timepoint lesion masks that all live in a common frame (e.g. the T0
frame produced by the longitudinal pipeline), this matches nodules across time
and reports how each one's volume / diameter / mass changes -- and can render a
per-nodule figure showing the lesion over time on three orthogonal projections.

Matching is by spatial correspondence in the common frame: the union of all
timepoint masks is connected-component labelled, and each timepoint's nodule is
assigned to the union component (the "track") it falls in.
"""
import os
from pathlib import Path

import numpy as np
import pandas as pd
import nibabel as nib

from src.modules.properties.nodule_properties import (
    NoduleProperties, connected_components_3d, _lesion_diameter, _voxel_spacing)


class NoduleTracker(object):
    def __init__(self, cfg=None):
        cfg = cfg or {}
        p = cfg.get('properties', cfg)
        self.min_size = p.get('min_size', 5)
        self.connectivity = p.get('connectivity', 26)
        self.properties = NoduleProperties(cfg)

    # ------------------------------------------------------------------ #
    def _label_all(self, timepoints):
        """Label each timepoint + the union; return maps, props and spacing."""
        labels = [t[0] for t in timepoints]
        lab_maps, props_by_id = [], []
        union = None
        spacing = None
        for label, mask_path, ct_path in timepoints:
            mimg = nib.load(str(mask_path))
            if union is None:
                union = np.zeros(mimg.shape, dtype=bool)
                spacing = _voxel_spacing(mimg.affine)
            mask = mimg.get_fdata() > 0
            union |= mask
            lab, _ = connected_components_3d(mask, self.connectivity, self.min_size)
            lab_maps.append(lab)
            props_by_id.append({d['nodule_id']: d
                                for d in self.properties.compute(mask_path, ct_path=ct_path)})
        track_lab, n_tracks = connected_components_3d(union, self.connectivity, self.min_size)
        return labels, lab_maps, props_by_id, track_lab, n_tracks, spacing

    def track(self, timepoints, time_days=None):
        """Match nodules across timepoints -> (long_df, summary_df)."""
        labels, lab_maps, props_by_id, track_lab, n_tracks, _ = self._label_all(timepoints)

        rows = []
        for ti, label in enumerate(labels):
            lab = lab_maps[ti]
            for cid, props in props_by_id[ti].items():
                tvals = track_lab[lab == cid]
                tvals = tvals[tvals > 0]
                if tvals.size == 0:
                    continue
                track_id = int(np.bincount(tvals).argmax())
                row = {'track_id': track_id, 'timepoint': label}
                row.update({k: props[k] for k in (
                    'volume_mm3', 'diameter_long_mm', 'diameter_short_mm',
                    'diameter_mm', 'mean_HU', 'mass_mg',
                    'centroid_x', 'centroid_y', 'centroid_z') if k in props})
                rows.append(row)
        long_df = pd.DataFrame(rows).sort_values(['track_id', 'timepoint']).reset_index(drop=True)
        summary_df = self._summarize(long_df, labels, time_days)
        return long_df, summary_df

    @staticmethod
    def _summarize(long_df, labels, time_days=None):
        order = {l: i for i, l in enumerate(labels)}
        summaries = []
        for track_id, g in long_df.groupby('track_id'):
            g = g.sort_values('timepoint', key=lambda s: s.map(order))
            base, last = g.iloc[0], g.iloc[-1]
            v0, v1 = base['volume_mm3'], last['volume_mm3']
            s = {
                'track_id': int(track_id),
                'n_timepoints': int(len(g)),
                'timepoints': ",".join(g['timepoint']),
                'baseline': base['timepoint'],
                'last': last['timepoint'],
                'volume_baseline_mm3': v0,
                'volume_last_mm3': v1,
                'volume_change_mm3': v1 - v0,
                'volume_change_pct': (100.0 * (v1 - v0) / v0) if v0 > 0 else np.nan,
                'diameter_baseline_mm': base['diameter_mm'],
                'diameter_last_mm': last['diameter_mm'],
                'diameter_change_mm': last['diameter_mm'] - base['diameter_mm'],
            }
            if 'mass_mg' in g.columns:
                m0, m1 = base['mass_mg'], last['mass_mg']
                s.update({'mass_baseline_mg': m0, 'mass_last_mg': m1, 'mass_change_mg': m1 - m0})
            if time_days is not None:
                dt = time_days.get(last['timepoint'], np.nan) - time_days.get(base['timepoint'], np.nan)
                s['vdt_days'] = (float(dt * np.log(2) / np.log(v1 / v0))
                                 if np.isfinite(dt) and dt > 0 and v0 > 0 and v1 > 0 and v1 != v0
                                 else np.nan)
            summaries.append(s)
        return pd.DataFrame(summaries)

    def track_to_files(self, timepoints, long_csv, summary_csv, time_days=None):
        long_df, summary_df = self.track(timepoints, time_days=time_days)
        long_df.to_csv(long_csv, index=False)
        summary_df.to_csv(summary_csv, index=False)
        return long_df, summary_df

    # ------------------------------------------------------------------ #
    def plot_tracks(self, timepoints, out_dir, min_timepoints=2,
                    padding=8, window=(-1000.0, 200.0), ref_masks=None,
                    masks=('pipeline', 'reference'), plot_3d=True, plot_lung=True,
                    time_days=None):
        """Render a figure per nodule track present in >= min_timepoints.

        Each figure has one row per timepoint. The first three columns are the
        Axial / Coronal / Sagittal CT slices through the nodule centroid, cropped
        to the nodule; an optional ``plot_3d`` column shows a 3D surface
        reconstruction of the pipeline nodule; an optional ``plot_lung`` column
        shows the whole (un-cropped) coronal CT slice with a marker at the nodule,
        to locate it within the lung. Overlaid mask contours are chosen by
        ``masks`` -- a subset of {'pipeline', 'reference'}: 'pipeline' draws the
        pipeline segmentation (red) and 'reference' draws the GT/reference (green)
        from ``ref_masks`` (a per-timepoint list of mask paths in the same common
        frame). Each row is annotated with volume (mm^3), average / max / min
        diameter (mm), mass (mg) and -- when ``time_days`` is given -- the elapsed
        time from baseline. Returns the list of PNG paths.
        """
        import matplotlib
        matplotlib.use('Agg')
        import matplotlib.pyplot as plt
        from matplotlib.lines import Line2D
        from mpl_toolkits.mplot3d import Axes3D  # noqa: F401  (registers '3d')

        out_dir = Path(out_dir)
        masks = {str(m).lower() for m in (masks or [])}
        show_pred = 'pipeline' in masks
        show_ref = 'reference' in masks
        time_days = time_days or {}

        labels, lab_maps, _, track_lab, n_tracks, spacing = self._label_all(timepoints)
        vox_vol = float(np.prod(spacing))
        cts = [nib.load(str(t[2])).get_fdata() for t in timepoints]   # raw HU
        clip = self.properties.clip_HU
        # Anatomical direction each array axis points toward (direction of
        # INCREASING voxel index), read from the common-frame affine -> the 3D
        # panel's axis labels. E.g. ('L','A','S') for LAS data.
        axcodes = nib.orientations.aff2axcodes(nib.load(str(timepoints[0][1])).affine)

        refs = None
        if show_ref and ref_masks is not None:
            refs = [(nib.load(str(rp)).get_fdata() > 0)
                    if rp and os.path.exists(str(rp)) else None for rp in ref_masks]
        has_ref = refs is not None and any(r is not None for r in refs)

        cols = ['axial', 'coronal', 'sagittal']
        if plot_lung:
            cols.append('lung')
        if plot_3d:
            cols.append('3d')
        ncols = len(cols)
        title_map = {'axial': 'Axial', 'coronal': 'Coronal', 'sagittal': 'Sagittal',
                     '3d': '3D', 'lung': 'Lung'}
        col_names = [title_map[k] for k in cols]

        paths = []
        for tid in range(1, n_tracks + 1):
            present = []
            for ti, label in enumerate(labels):
                reg = (lab_maps[ti] > 0) & (track_lab == tid)
                if reg.sum() > 0:
                    present.append((ti, label, reg))
            if len(present) < min_timepoints:
                continue

            nrows = len(present)
            base_label = present[0][1]
            # Shared physical box side (mm) for every 3D panel of this track, so
            # the nodule's growth is visible across timepoints (all timepoints are
            # in the same common frame, hence directly comparable). Sized to the
            # largest timepoint's bounding box, with a small margin.
            axis_len = 0.0
            for _ti, _lab, _reg in present:
                c = np.argwhere(_reg)
                ext = (c.max(0) - c.min(0) + 1).astype(float) * spacing
                axis_len = max(axis_len, float(ext.max()))
            axis_len *= 1.1
            fig = plt.figure(figsize=(3 * ncols, 3 * nrows))
            axes = np.empty((nrows, ncols), dtype=object)
            for r in range(nrows):
                for c in range(ncols):
                    proj = '3d' if cols[c] == '3d' else None
                    axes[r, c] = fig.add_subplot(nrows, ncols, r * ncols + c + 1,
                                                 projection=proj)

            for r, (ti, label, reg) in enumerate(present):
                coords = np.argwhere(reg)
                cen = coords.mean(0).round().astype(int)
                mn, mx = coords.min(0), coords.max(0)
                vol = float(coords.shape[0]) * vox_vol
                _, dlong, dshort, diam = _lesion_diameter(reg, spacing[0], spacing[1])
                hu = float(np.clip(cts[ti][reg], clip[0], clip[1]).mean())
                mass = vol * (hu / 1000.0 + 1.0)
                ct_disp = np.clip(cts[ti], window[0], window[1])
                ref = refs[ti] if refs is not None else None
                pred = reg if show_pred else None
                # Axial: fix z (axis 2) -> plane (x, y); Coronal: fix y -> (x, z);
                # Sagittal: fix x -> (y, z).
                self._panel(axes[r, 0], ct_disp[:, :, cen[2]],
                            (pred[:, :, cen[2]] if pred is not None else None),
                            (mn[0], mx[0], mn[1], mx[1]), padding,
                            ref2d=(ref[:, :, cen[2]] if ref is not None else None))
                self._panel(axes[r, 1], ct_disp[:, cen[1], :],
                            (pred[:, cen[1], :] if pred is not None else None),
                            (mn[0], mx[0], mn[2], mx[2]), padding,
                            ref2d=(ref[:, cen[1], :] if ref is not None else None))
                self._panel(axes[r, 2], ct_disp[cen[0], :, :],
                            (pred[cen[0], :, :] if pred is not None else None),
                            (mn[1], mx[1], mn[2], mx[2]), padding,
                            ref2d=(ref[cen[0], :, :] if ref is not None else None))
                if plot_3d:
                    self._panel3d(axes[r, cols.index('3d')], reg, mn, mx, padding,
                                  spacing, axis_len=axis_len, axcodes=axcodes)
                if plot_lung:
                    # whole coronal CT slice (fix y at the nodule centroid) with a
                    # marker at the nodule's (x, z) position -> anatomical context.
                    self._panel_lung(axes[r, cols.index('lung')],
                                     ct_disp[:, cen[1], :], (cen[0], cen[2]))

                lines = [label]
                if label in time_days and base_label in time_days:
                    dt = time_days[label] - time_days[base_label]
                    lines.append(f"$\\Delta t$={dt:.0f} d")
                lines += [f"V={vol:.0f} mm$^3$",
                          f"$D_{{avg}}$={diam:.1f} mm",
                          f"$D_{{max}}$={dlong:.1f} mm",
                          f"$D_{{min}}$={dshort:.1f} mm",
                          f"mass={mass:.0f} mg"]
                axes[r, 0].set_ylabel("\n".join(lines), fontsize=12, rotation=0,
                                      ha='right', va='center', labelpad=44)
            for c, name in enumerate(col_names):
                axes[0, c].set_title(name)
            fig.suptitle(f"Nodule track {tid} over time")
            handles = []
            if show_pred:
                handles.append(Line2D([0], [0], color='r', lw=1.5, label='pipeline'))
            if show_ref and has_ref:
                handles.append(Line2D([0], [0], color='lime', lw=1.5, label='reference'))
            if handles:
                fig.legend(handles=handles, loc='lower center', ncol=len(handles),
                           frameon=False)
            fig.tight_layout(rect=[0.03, 0.03, 1, 0.98])
            p = out_dir / f"growth_track_{tid}.png"
            fig.savefig(str(p), dpi=120, bbox_inches='tight')
            plt.close(fig)
            paths.append(str(p))
        return paths

    @staticmethod
    def _panel(ax, img2d, msk2d, bbox2d, pad, ref2d=None):
        a0, a1, b0, b1 = bbox2d
        A0, A1 = max(0, a0 - pad), min(img2d.shape[0], a1 + pad + 1)
        B0, B1 = max(0, b0 - pad), min(img2d.shape[1], b1 + pad + 1)
        sub = img2d[A0:A1, B0:B1]
        ax.imshow(sub.T, cmap='gray', origin='lower')
        if msk2d is not None:
            smsk = msk2d[A0:A1, B0:B1]
            if smsk.any():
                ax.contour(smsk.T.astype(float), levels=[0.5], colors='r', linewidths=1.0)
        if ref2d is not None:
            sref = ref2d[A0:A1, B0:B1]
            if sref.any():
                ax.contour(sref.T.astype(float), levels=[0.5], colors='lime', linewidths=1.0)
        ax.set_xticks([])
        ax.set_yticks([])

    @staticmethod
    def _panel_lung(ax, ct2d, marker):
        """Whole (un-cropped) CT slice with a marker at the nodule position."""
        ax.imshow(ct2d.T, cmap='gray', origin='lower')
        mx, my = marker
        ax.plot(mx, my, marker='o', markersize=8, markerfacecolor='none',
                markeredgecolor='r', markeredgewidth=1.5)
        ax.plot(mx, my, marker='+', markersize=10, color='r', markeredgewidth=1.0)
        ax.set_xticks([])
        ax.set_yticks([])

    @staticmethod
    def _panel3d(ax, reg, mn, mx, pad, spacing, axis_len=None, axcodes=None):
        """3D surface reconstruction of the pipeline nodule (cropped to bbox).

        If ``axis_len`` (a physical box side in mm) is given, all axes are set to
        that same span, centred on the nodule -- so panels rendered for different
        timepoints of a track share one scale and growth is directly visible.

        The x/y/z axes are the image array axes scaled to mm. ``axcodes`` (from
        ``nibabel.aff2axcodes``) gives the anatomical direction each axis points
        toward (increasing index) and is used to label the box.
        """
        sl = tuple(slice(max(0, int(mn[k]) - pad), min(reg.shape[k], int(mx[k]) + pad + 1))
                   for k in range(3))
        vol = reg[sl]
        if vol.sum() == 0:
            ax.set_axis_off()
            return
        drawn = False
        try:
            from skimage.measure import marching_cubes
            verts, faces, _, _ = marching_cubes(
                vol.astype(np.float32), level=0.5, spacing=tuple(float(s) for s in spacing))
            ax.plot_trisurf(verts[:, 0], verts[:, 1], faces, verts[:, 2],
                            color='r', alpha=0.6, linewidth=0, antialiased=True)
            # centre (mm) of the reconstructed surface within the crop
            centre = 0.5 * (verts.min(0) + verts.max(0))
            drawn = True
        except Exception:
            pass
        if not drawn:
            ax.voxels(vol, facecolors=(1.0, 0.0, 0.0, 0.6), edgecolor=None)
            # voxels render in index units; centre in mm-equivalent index space
            centre = np.array([0.5 * vol.shape[k] * float(spacing[k]) for k in range(3)])

        if axis_len:
            half = 0.5 * float(axis_len)
            # marching-cubes verts are in mm; voxels are in index units, so map the
            # shared mm span back to index units for the fallback.
            scale = np.ones(3) if drawn else np.array([float(spacing[k]) for k in range(3)])
            c = centre if drawn else centre / scale
            h = half / scale
            ax.set_xlim(c[0] - h[0], c[0] + h[0])
            ax.set_ylim(c[1] - h[1], c[1] + h[1])
            ax.set_zlim(c[2] - h[2], c[2] + h[2])
            ax.set_box_aspect([1.0, 1.0, 1.0] if drawn
                              else [float(s) for s in spacing])
        else:
            try:
                ax.set_box_aspect([vol.shape[k] * float(spacing[k]) for k in range(3)])
            except Exception:
                pass

        # anatomical axis labels (direction of increasing coordinate) + a fixed
        # camera so all timepoints of a track are viewed from the same angle.
        full = {'L': 'L (left)', 'R': 'R (right)', 'A': 'A (ant.)', 'P': 'P (post.)',
                'S': 'S (sup.)', 'I': 'I (inf.)'}
        codes = axcodes if axcodes else ('i0', 'i1', 'i2')
        ax.set_xlabel(full.get(codes[0], codes[0]) + " →", fontsize=7, labelpad=-9)
        ax.set_ylabel(full.get(codes[1], codes[1]) + " →", fontsize=7, labelpad=-9)
        ax.set_zlabel(full.get(codes[2], codes[2]) + " →", fontsize=7, labelpad=-9)
        ax.view_init(elev=20, azim=-60)
        ax.set_xticks([])
        ax.set_yticks([])
        ax.set_zticks([])
        ax.grid(False)
