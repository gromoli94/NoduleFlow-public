"""Longitudinal NoduleFlow pipeline with candidate propagation.

Given several timepoints of the same patient, the pipeline:

  1. detects + classifies candidates at each timepoint (own image space),
  2. brings every timepoint into a common frame by registering the follow-ups
     onto the fixed timepoint (default T0) -- reusing already-registered niftis
     when present, otherwise registering them,
  3. propagates candidate bounding boxes between timepoints according to a
     strategy, unioning them into a per-timepoint prompt mask,
  4. segments each timepoint (LesionLocator) from its (propagated) prompt mask.

Propagation strategies (as specified):
  S0  no propagation            -- each timepoint uses only its own candidates
  S1  from previous times only  -- timepoint j also gets candidates from i <= j
  S2  across all times          -- timepoint j gets candidates from every i

(Note: the source repo's S1/S2 labels are swapped relative to these.)
"""
import os
import logging
from pathlib import Path

import numpy as np
import nibabel as nib
import torch

from src.modules.detection.detector import Detector
from src.modules.segmentation.segmenter import Segmenter
from src.modules.registration.registrator import Registrator
from src.evaluation.evaluator import Evaluator
from src.io.bbox_mask import detections_to_bbox_mask
from src.modules.properties.nodule_properties import NoduleProperties
from src.modules.properties.tracking import NoduleTracker


class LongitudinalPipeline(object):
    def __init__(self, cfg, logFile=None, checkIfExists=False):
        self.cfg = cfg
        self.logFile = logFile
        self.checkIfExists = checkIfExists
        lon = cfg.get('longitudinal', {})
        self.strategy = lon.get('strategy', 'S1').upper()
        if self.strategy not in ('S0', 'S1', 'S2'):
            raise ValueError(f"Unknown strategy '{self.strategy}'. Use S0, S1 or S2.")
        self.fixed_time = cfg.get('registration', {}).get('fixed_time', 'T0')

        pipe = cfg.get('pipeline', {})
        self.modality = pipe.get('modality', 'box')
        self.max_prompts = pipe.get('max_prompts', 5)
        self.output_tag = pipe.get('output_tag', 'pipeline')
        self.bbox_tag = pipe.get('bbox_tag', 'BB')
        self.compute_properties = pipe.get('compute_properties', False)
        self.growth_report = lon.get('growth_report', False)
        self.time_days = lon.get('time_days', None)
        self.plot_growth = lon.get('plot_growth', False)
        self.plot_min_timepoints = lon.get('plot_min_timepoints', 2)
        self.plot_masks = lon.get('plot_masks', ['pipeline', 'reference'])
        self.plot_3d = lon.get('plot_3d', True)
        self.plot_lung = lon.get('plot_lung', True)

        # The pipeline paints boxes, so the detector must emit box coordinates.
        cfg.setdefault('detector', {})['output'] = 'box'
        cfg.setdefault('lesionlocator', {})['nodule_tag'] = self.output_tag

        self.detector = Detector(cfg, logFile=logFile)
        self.segmenter = Segmenter(cfg, logFile=logFile, checkIfExists=checkIfExists)
        self.evaluator = Evaluator(cfg)
        self.properties = NoduleProperties(cfg) if self.compute_properties else None
        self.tracker = NoduleTracker(cfg) if self.growth_report else None
        # Registration only needed when propagating.
        self.registrator = Registrator(cfg, logFile=logFile) if self.strategy != 'S0' else None

    # ------------------------------------------------------------------ #
    def _select(self, detections):
        detections = sorted(
            detections,
            key=lambda d: d.get('nodule_score', 0.0) * d.get('lesion_score', 1.0),
            reverse=True)
        if self.max_prompts and self.max_prompts > 0:
            detections = detections[:self.max_prompts]
        return detections

    def _free_gpu(self):
        """Move the parent GPU models off the device so the LesionLocator
        subprocess (nnU-Net) has the GPU to itself for segmentation."""
        try:
            self.detector.detector.to('cpu')
            if self.detector.classifier is not None:
                self.detector.classifier.model.to('cpu')
            if self.registrator is not None:
                self.registrator.net.to('cpu')
        except Exception as e:
            logging.warning(f"Could not move models to CPU: {e}")
        if torch.cuda.is_available():
            torch.cuda.empty_cache()

    def _donors(self, j, n):
        """Indices whose candidates propagate into timepoint j (own index included)."""
        if self.strategy == 'S0':
            return [j]
        if self.strategy == 'S1':          # own + previous
            return [i for i in range(n) if i <= j]
        return list(range(n))              # S2: all times

    @staticmethod
    def _union_masks(mask_paths, reference_nifti, out_path):
        ref = nib.load(str(reference_nifti))
        combined = np.zeros(ref.shape, dtype=np.uint8)
        for p in mask_paths:
            if p is None or not os.path.exists(p):
                continue
            combined |= (nib.load(str(p)).get_fdata() > 0).astype(np.uint8)
        hdr = ref.header.copy()
        hdr.set_data_dtype(np.uint8)
        nib.save(nib.Nifti1Image(combined, affine=ref.affine, header=hdr), str(out_path))
        return str(out_path)

    # ------------------------------------------------------------------ #
    def Run(self, timepoints, outputFolder, gt_masks=None):
        """Run the longitudinal pipeline.

        Parameters
        ----------
        timepoints : list of (label, nifti_path), ordered in time (e.g. T0, T1, T2).
        gt_masks : dict {label: gt_mask_path} or None -- for optional evaluation.

        Returns
        -------
        dict {label: {mask, prompt_mask, image, detections, metrics?}}
        """
        outputFolder = Path(outputFolder)
        outputFolder.mkdir(parents=True, exist_ok=True)
        labels = [t[0] for t in timepoints]
        niftis = [os.path.abspath(str(t[1])) for t in timepoints]
        n = len(timepoints)
        gt_masks = gt_masks or {}

        fixed_idx = labels.index(self.fixed_time) if self.fixed_time in labels else 0
        fixed_nifti = niftis[fixed_idx]

        # 1. detect + select per timepoint (own space); paint own BB mask
        own_bb, own_dets = [], []
        for label, nifti in zip(labels, niftis):
            print(f"[longitudinal] detecting candidates at {label} ...")
            dets = self._select(self.detector.Detect([nifti]).get(nifti, []))
            own_dets.append(dets)
            stem = Path(nifti).name[:-7]
            bb_path = outputFolder / f"{stem}_{self.bbox_tag}.nii.gz"
            detections_to_bbox_mask(dets, nifti, bb_path, instance=False)
            own_bb.append(str(bb_path))
            print(f"[longitudinal]   {label}: {len(dets)} candidate(s)")

        # gt_common[label] -> GT mask in the same space as that timepoint's output.
        # For S0 the output is in own space, so GT is used as-is; for S1/S2 the
        # output is in the fixed (T0) frame, so each moving GT is warped there
        # with the same phi_AB field used for the candidates.
        gt_common = {label: gt_masks.get(label) for label in labels}

        # 2. registration + bring BB masks into a common frame (skip for S0)
        if self.strategy == 'S0':
            bb_common = list(own_bb)          # own space
            seg_targets = list(niftis)        # segment original images
            ref_for_union = niftis            # per-timepoint own grid
            field_invs = [None] * n           # no registration -> already own space
        else:
            bb_common = [None] * n
            seg_targets = [None] * n
            field_invs = [None] * n            # phi_BA per moving timepoint (fixed->own)
            # Lesion masks to exclude from the field estimation (register_with_mask).
            use_mask = getattr(self.registrator, 'register_with_mask', False)
            fixed_mask = gt_masks.get(self.fixed_time) if use_mask else None
            if use_mask and fixed_mask is None:
                logging.warning("register_with_mask=True but no mask for the fixed "
                                f"timepoint '{self.fixed_time}'; registering on whole image.")
            for i, (label, nifti) in enumerate(zip(labels, niftis)):
                if i == fixed_idx:
                    bb_common[i] = own_bb[i]   # already in fixed frame
                    seg_targets[i] = nifti
                    continue
                movingMasks = [(own_bb[i], self.bbox_tag)]
                # warp this timepoint's GT into the fixed frame too (nearest-nb)
                if gt_masks.get(labels[i]):
                    movingMasks.append((gt_masks[labels[i]], 'gt'))
                moving_mask = gt_masks.get(labels[i]) if use_mask else None
                res = self.registrator.RegisterOrLoad(
                    fixed_nifti, nifti, outputFolder,
                    movingMasks=movingMasks,
                    fixedMask=fixed_mask, movingMask=moving_mask,
                    checkIfExists=True)
                bb_common[i] = res['warped_masks'][self.bbox_tag]
                seg_targets[i] = res['warped_image']
                field_invs[i] = res['field_inv']
                if 'gt' in res.get('warped_masks', {}):
                    gt_common[labels[i]] = res['warped_masks']['gt']
            ref_for_union = [fixed_nifti] * n  # everything on the fixed grid

        # 3. propagate (union donors) into a per-timepoint prompt mask
        prompt_masks = []
        for j, (label, nifti) in enumerate(zip(labels, niftis)):
            donors = self._donors(j, n)
            stem = Path(seg_targets[j]).name[:-7]
            prompt_path = outputFolder / f"{stem}_{self.bbox_tag}_{self.strategy}.nii.gz"
            self._union_masks([bb_common[i] for i in donors], ref_for_union[j], prompt_path)
            prompt_masks.append(str(prompt_path))
            print(f"[longitudinal] {label}: prompt from donors {[labels[i] for i in donors]}")

        # Free the detection/registration GPU models before the segmenter
        # subprocess (LesionLocator's nnU-Net) claims the device.
        self._free_gpu()

        # 4. segment each timepoint with its (propagated) prompt mask
        results = {}
        for j, label in enumerate(labels):
            target = seg_targets[j]
            self.segmenter.Segment(niftis=[target], annFiles=[prompt_masks[j]],
                                   outputFolder=outputFolder, modality=self.modality)
            stem = Path(target).name[:-7]
            out_mask = outputFolder / f"{stem}_{self.output_tag}.nii.gz"
            entry = {'mask': str(out_mask), 'prompt_mask': prompt_masks[j],
                     'image': target, 'detections': own_dets[j],
                     'warped_bb': bb_common[j]}  # this timepoint's own candidates in the common frame
            # Evaluation in the OUTPUT space (own space for S0; fixed frame for
            # S1/S2, against the GT warped into that frame).
            if gt_common.get(label):
                cand_csv = outputFolder / f"{stem}_candidates.csv"
                glob_csv = outputFolder / f"{stem}_global.csv"
                _, metrics = self.evaluator.evaluate_to_files(
                    out_mask, gt_common[label], own_dets[j], cand_csv, glob_csv)
                entry.update({'candidates_csv': str(cand_csv),
                              'global_csv': str(glob_csv),
                              'gt_used': gt_common[label], 'metrics': metrics})

            # Own-space output: for S1/S2 follow-ups the output lives in the fixed
            # frame, so warp it back to this timepoint's own grid (phi_BA) so
            # own-space evaluation and properties are possible.
            own_stem = Path(niftis[j]).name[:-7]
            own_mask = None
            if field_invs[j] is not None:
                own_mask = outputFolder / f"{own_stem}_{self.output_tag}_own.nii.gz"
                self.registrator.WarpBack(out_mask, niftis[j], field_invs[j],
                                          own_mask, nearest=True)
                entry['mask_own'] = str(own_mask)

            # Own-space evaluation against the original own-space GT.
            if gt_masks.get(label):
                if own_mask is not None:
                    cand_own = outputFolder / f"{own_stem}_candidates_own.csv"
                    glob_own = outputFolder / f"{own_stem}_global_own.csv"
                    _, metrics_own = self.evaluator.evaluate_to_files(
                        own_mask, gt_masks[label], own_dets[j], cand_own, glob_own)
                    entry.update({'candidates_csv_own': str(cand_own),
                                  'global_csv_own': str(glob_own),
                                  'metrics_own': metrics_own})
                elif 'metrics' in entry:
                    # already in own space (S0, or the fixed timepoint itself)
                    entry['metrics_own'] = entry['metrics']

            # Optional per-nodule properties CSV(s) alongside the segmentation.
            if self.properties is not None:
                props_csv = outputFolder / f"{stem}_properties.csv"
                self.properties.compute_to_csv(out_mask, props_csv, ct_path=target)
                entry['properties_csv'] = str(props_csv)
                if own_mask is not None:
                    props_own = outputFolder / f"{own_stem}_properties_own.csv"
                    self.properties.compute_to_csv(own_mask, props_own, ct_path=niftis[j])
                    entry['properties_csv_own'] = str(props_own)

            results[label] = entry
            print(f"[longitudinal] {label}: segmentation -> {out_mask}")

        # 5. cross-timepoint growth report (needs a common frame -> S1/S2)
        if self.tracker is not None:
            if self.strategy == 'S0':
                logging.warning("growth_report requested but strategy=S0 has no common "
                                "frame; skipping (use S1/S2).")
                print("[longitudinal] growth report skipped for S0 (no common frame).")
            else:
                # all timepoint outputs live in the fixed frame (results[*]['mask'],
                # with the matching fixed-frame CT in results[*]['image']).
                tps = [(label, results[label]['mask'], results[label]['image'])
                       for label in labels]
                long_csv = outputFolder / "growth_long.csv"
                summary_csv = outputFolder / "growth_summary.csv"
                long_df, summary_df = self.tracker.track_to_files(
                    tps, long_csv, summary_csv, time_days=self.time_days)
                results['growth'] = {'long_csv': str(long_csv),
                                     'summary_csv': str(summary_csv),
                                     'n_tracks': int(summary_df['track_id'].nunique())
                                     if len(summary_df) else 0}
                print(f"[longitudinal] growth report: {summary_csv} "
                      f"({results['growth']['n_tracks']} track(s))")
                # per-nodule growth figures (lesion over time, 3 projections)
                if self.plot_growth:
                    # reference masks in the common frame (warped GT for follow-ups)
                    ref_masks = [gt_common.get(label) for label in labels]
                    plots = self.tracker.plot_tracks(
                        tps, outputFolder, min_timepoints=self.plot_min_timepoints,
                        ref_masks=ref_masks, masks=self.plot_masks,
                        plot_3d=self.plot_3d, plot_lung=self.plot_lung,
                        time_days=self.time_days)
                    results['growth']['plots'] = plots
                    print(f"[longitudinal] growth plots: {len(plots)} figure(s)")
        return results
