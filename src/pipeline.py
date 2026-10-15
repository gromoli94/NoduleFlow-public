"""Single-scan NoduleFlow pipeline.

Given one CT NIfTI, the pipeline:

  1. runs the Detector (RetinaNet detection + ResNet50 classification, with
     optional ensemble false-positive reduction) to get candidate boxes,
  2. keeps the strongest candidates and paints them into a bounding-box mask,
  3. feeds that mask as a prompt to the Segmenter (LesionLocator), which
     produces the final lesion segmentation.

Output: a single mask NIfTI written to the chosen output folder.
"""
import os
import logging
from pathlib import Path

from src.modules.detection.detector import Detector
from src.modules.segmentation.segmenter import Segmenter
from src.io.bbox_mask import detections_to_bbox_mask
from src.evaluation.evaluator import Evaluator
from src.modules.properties.nodule_properties import NoduleProperties


class Pipeline(object):
    def __init__(self, cfg, logFile=None, checkIfExists=False):
        self.cfg = cfg
        self.logFile = logFile
        pipe = cfg.get('pipeline', {})
        self.modality = pipe.get('modality', 'box')          # box | point prompts
        self.max_prompts = pipe.get('max_prompts', 5)        # keep top-K (0 = all)
        self.output_tag = pipe.get('output_tag', 'pipeline')
        self.bbox_tag = pipe.get('bbox_tag', 'BB')           # intermediate BB mask
        self.compute_properties = pipe.get('compute_properties', False)

        # The pipeline paints boxes, so the detector must emit box coordinates.
        cfg.setdefault('detector', {})['output'] = 'box'
        # Make the segmenter's output carry the pipeline tag.
        cfg.setdefault('lesionlocator', {})['nodule_tag'] = self.output_tag

        self.detector = Detector(cfg, logFile=logFile)
        self.segmenter = Segmenter(cfg, logFile=logFile, checkIfExists=checkIfExists)
        self.evaluator = Evaluator(cfg)
        self.properties = NoduleProperties(cfg) if self.compute_properties else None

    def _select(self, detections):
        """Keep the strongest detections as prompts (by combined score)."""
        detections = sorted(
            detections,
            key=lambda d: d.get('nodule_score', 0.0) * d.get('lesion_score', 1.0),
            reverse=True,
        )
        if self.max_prompts and self.max_prompts > 0:
            detections = detections[:self.max_prompts]
        return detections

    def Run(self, nifti, outputFolder, gt_mask=None):
        """Run the full pipeline on a single CT NIfTI.

        If `gt_mask` (a ground-truth lesion mask) is given, also writes an
        evaluation next to the output mask: a per-candidate CSV and a global
        metrics CSV.

        Returns a dict with keys 'mask' (always) and, when evaluated,
        'candidates_csv', 'global_csv' and 'metrics'.
        """
        nifti = os.path.abspath(str(nifti))
        outputFolder = Path(outputFolder)
        outputFolder.mkdir(parents=True, exist_ok=True)
        stem = Path(nifti).name[:-7]  # strip .nii.gz

        # 1. detection (+ classification + optional ensemble)
        print(f"[pipeline] detecting candidates in {nifti} ...")
        detections = self.detector.Detect([nifti]).get(nifti, [])
        print(f"[pipeline] {len(detections)} candidate detection(s) before selection")

        # 2. select the strongest candidates
        detections = self._select(detections)
        print(f"[pipeline] {len(detections)} candidate(s) kept as prompts "
              f"(max_prompts={self.max_prompts})")
        if not detections:
            logging.warning("Pipeline: no detections to prompt the segmenter with.")

        # 3. paint boxes into a mask that prompts the segmenter
        bbMaskPath = outputFolder / f"{stem}_{self.bbox_tag}.nii.gz"
        detections_to_bbox_mask(detections, nifti, bbMaskPath, instance=False)
        print(f"[pipeline] wrote bounding-box prompt mask: {bbMaskPath}")

        # 4. segment with the prompt mask
        self.segmenter.Segment(niftis=[nifti], annFiles=[str(bbMaskPath)],
                               outputFolder=outputFolder, modality=self.modality)

        # 5. final output
        outPath = outputFolder / f"{stem}_{self.output_tag}.nii.gz"
        print(f"[pipeline] final segmentation: {outPath}")
        logging.info(f"Pipeline output: {outPath}")
        result = {'mask': str(outPath)}

        # 6a. optional per-nodule properties CSV (uses the input CT for HU/mass)
        if self.properties is not None:
            props_csv = outputFolder / f"{stem}_properties.csv"
            self.properties.compute_to_csv(outPath, props_csv, ct_path=nifti)
            print(f"[pipeline] wrote nodule properties: {props_csv}")
            result['properties_csv'] = str(props_csv)

        # 6. optional evaluation against ground truth
        if gt_mask is not None:
            candidates_csv = outputFolder / f"{stem}_candidates.csv"
            global_csv = outputFolder / f"{stem}_global.csv"
            _, metrics = self.evaluator.evaluate_to_files(
                outPath, gt_mask, detections, candidates_csv, global_csv)
            print(f"[pipeline] wrote candidate metrics: {candidates_csv}")
            print(f"[pipeline] wrote global metrics:    {global_csv}")
            print(f"[pipeline] sensitivity={metrics['sensitivity']:.3f} "
                  f"precision={metrics['precision']:.3f} "
                  f"volumetric_dice={metrics['volumetric_dice']:.3f}")
            result.update({'candidates_csv': str(candidates_csv),
                           'global_csv': str(global_csv),
                           'metrics': metrics})
        return result
