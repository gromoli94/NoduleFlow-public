import os
import json
import logging
import tempfile

import numpy as np
import torch
import nibabel as nib
from monai.data import DataLoader, Dataset, load_decathlon_datalist
from monai.data.utils import no_collation
from monai.transforms import ScaleIntensityRanged
from monai.apps.detection.networks.retinanet_detector import RetinaNetDetector
from monai.apps.detection.utils.anchor_utils import AnchorGeneratorWithAnchorShape

from src.utils.generate_transforms import generate_detection_inference_transform
from src.modules.classification.classifier import Classifier
from src.modules.detection.ensemble import EnsembleReducer

## DETECTOR MODULE
## Two-stage lung-nodule detection:
##   1. RetinaNet detector (MONAI, TorchScript weights) proposes candidate boxes.
##   2. Optional ResNet50 classifier (false-positive reduction) rescores each box
##      -- delegated to src.modules.classification.classifier.Classifier.
## Returns per-scan detections as box or point coordinates in the input image's
## voxel space.

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")


class Detector(object):
    def __init__(self, cfg, logFile=None):
        self.cfg = cfg
        self.logFile = logFile
        det = cfg.get('detector', {})
        self.detection_patch_size = [192, 192, 80]
        self.detectorName = det.get('model', 'DLCSD-mD')
        self.maxDetectionsPerImage = det.get('max_detections', 250)
        self.nms_threshold = det.get('nms_threshold', 0.22)
        self.lesionThrs = det.get('score_threshold', 0.0)
        self.output_mode = det.get('output', 'box')  # 'box' or 'point'
        self.weights_dir = cfg.get('network_weights_dir', 'network_weights')

        classifierCfg = det.get('classifier', {})
        self.use_classifier = classifierCfg.get('enabled', True)
        self.noduleThrs = classifierCfg.get('threshold', 0.0)

        if self.output_mode not in ('box', 'point'):
            raise ValueError(f"Unsupported output '{self.output_mode}'. Use 'box' or 'point'.")

        # Ensemble false-positive reduction: merge spatially overlapping
        # candidates into single detections (optional, off by default).
        self.ensemble = EnsembleReducer(cfg=det.get('ensemble', {}))

        self.detector = self.GetBounder()
        # Delegate false-positive reduction to the classification module.
        if self.use_classifier:
            self.classifier = Classifier({
                'classifier': classifierCfg,
                'network_weights_dir': self.weights_dir,
            }, logFile=self.logFile)
        else:
            self.classifier = None

    def GetBounder(self):
        logging.info("Loading RetinaNetDetector for bounding box detection...")
        print("Loading RetinaNetDetector for bounding box detection...")
        networkWeights = f"{self.detectorName}.pt"
        modelPath = os.path.join(self.weights_dir, self.detectorName, networkWeights)
        net = torch.jit.load(modelPath).to(device)
        net.to(device)
        # build anchors (must match train config)
        anchor_generator = AnchorGeneratorWithAnchorShape(
            feature_map_scales=[2**l for l in range(len([1, 2]) + 1)],  # 3 levels → [1,2,4]
            base_anchor_shapes=[[6, 8, 4],
                                [8, 6, 5],
                                [10, 10, 6]],)
        # build detector
        detector = RetinaNetDetector(network=net, anchor_generator=anchor_generator, debug=False)
        # set detection parameters (must match train config)
        detector.set_box_selector_parameters(
            score_thresh=0.02,  # keep boxes with >=2% probability of containing a lesion
            topk_candidates_per_level=1000,
            detections_per_img=self.maxDetectionsPerImage,
            nms_thresh=self.nms_threshold,)
        detector.set_sliding_window_inferer(
            roi_size=self.detection_patch_size,
            overlap=0.22,
            sw_batch_size=1,
            mode="gaussian",
            device=device,)
        detector.to(device).eval()
        logging.info("RetinaNetDetector loaded.")
        print("RetinaNetDetector loaded.")
        return detector

    @staticmethod
    def DefineDetectionDataTransforms():
        intensity_transform = ScaleIntensityRanged(
            keys=["image"],
            a_min=-1024,
            a_max=300.0,
            b_min=0.0,
            b_max=1.0,
            clip=True,
        )
        inference_transforms, post_transforms = generate_detection_inference_transform(
            "image",
            "pred_box",
            "pred_label",
            "pred_score",
            "cccwhd",  # coordinate format
            intensity_transform,
            affine_lps_to_ras=False,
            amp=True,
        )
        return inference_transforms, post_transforms

    @staticmethod
    def CreateDataList(niftis, data_list_key="data", outputDir=None, outputFileName="niftiList.json"):
        list_dict = {data_list_key: []}
        list_dict[data_list_key].extend([{"image": p} for p in niftis if 'mask' not in p])
        outputDir = outputDir or tempfile.mkdtemp(prefix="det_")
        os.makedirs(outputDir, exist_ok=True)
        listFilePath = os.path.join(outputDir, outputFileName)
        with open(listFilePath, "w") as f:
            json.dump(list_dict, f, indent=2)
        return listFilePath

    def CreateDetectionDataLoader(self, niftis, data_list_key="data", batch_size=10):
        listFilePath = self.CreateDataList(niftis, data_list_key=data_list_key)
        detection_transforms, post_transforms = self.DefineDetectionDataTransforms()
        data = load_decathlon_datalist(listFilePath,
                                        data_list_key=data_list_key,
                                        base_dir=None,)  # absolute paths already specified
        ds = Dataset(data=data, transform=detection_transforms,)
        detection_loader = DataLoader(ds,
                                      batch_size=batch_size,
                                      num_workers=4,
                                      pin_memory=torch.cuda.is_available(),
                                      collate_fn=no_collation,)
        return data, detection_loader, post_transforms

    @staticmethod
    def _box_to_voxel_coords(box, shape):
        """Convert a raw detector box [z0,x0,y0,z1,x1,y1] to voxel-index corners
        (x_min,x_max,y_min,y_max,z_min,z_max) in the original image array.

        The x-axis is flipped because detection runs on the RAS-oriented volume
        while the returned coordinates index the input image array directly.
        """
        z_min_f, x_min_f, y_min_f, z_max_f, x_max_f, y_max_f = box
        x_max = shape[0] - 1 - int(np.floor(x_min_f))
        x_min = shape[0] - 1 - int(np.ceil(x_max_f))
        y_min = int(np.floor(y_min_f))
        y_max = int(np.ceil(y_max_f))
        z_min = int(np.floor(z_min_f))
        z_max = int(np.ceil(z_max_f))
        return x_min, x_max, y_min, y_max, z_min, z_max

    def _format_detection(self, d):
        """Return a detection in the configured output mode ('box' or 'point'),
        preserving score fields and any extra keys (e.g. 'n_merged')."""
        extra = {k: v for k, v in d.items() if k not in
                 ("x_min", "x_max", "y_min", "y_max", "z_min", "z_max")}
        if self.output_mode == "point":
            out = dict(extra)
            out.update({
                "x": int(round((d["x_min"] + d["x_max"]) / 2)),
                "y": int(round((d["y_min"] + d["y_max"]) / 2)),
                "z": int(round((d["z_min"] + d["z_max"]) / 2)),
            })
            return out
        return d

    def Detect(self, niftis):
        """Run detection (+ optional classification) on a list of NIfTI files.

        Returns
        -------
        dict mapping each nifti path -> list of detections. Each detection is a
        dict with 'nodule_score', 'lesion_score' and, depending on output mode:
          - 'box'  : {'x_min','x_max','y_min','y_max','z_min','z_max'}
          - 'point': {'x','y','z'} (box centre, voxel indices)
        """
        def to_numpy(tensor):
            return tensor.to(torch.float32).cpu().detach().numpy()

        # Absolute paths keep MONAI's loader and the nifti_to_idx lookup consistent.
        niftis = [os.path.abspath(str(n)) for n in niftis]
        nifti_to_idx = {n: i for i, n in enumerate(niftis)}
        _, detection_loader, _ = self.CreateDetectionDataLoader(niftis)
        boxes_per_nifti = [[] for _ in niftis]
        scores_per_nifti = [[] for _ in niftis]

        with torch.no_grad():
            for batch in detection_loader:
                for item in batch:
                    img_tensor = item["image"]
                    nifti_file = item["image_meta_dict"]["filename_or_obj"]
                    index = nifti_to_idx[nifti_file]
                    try:
                        print(f"Processing {nifti_file}...")
                        logging.info(f"Processing {nifti_file}...")
                        use_inferer = img_tensor.numel() >= np.prod(self.detection_patch_size)
                        inputDet = img_tensor.unsqueeze(1).to(device)
                        inputDet = inputDet.permute(0, 1, 4, 2, 3)
                        with torch.amp.autocast('cuda'):
                            outputDet = self.detector(inputDet, use_inferer=use_inferer)
                        boxes = to_numpy(outputDet[0][self.detector.target_box_key])
                        scores = to_numpy(outputDet[0][self.detector.pred_score_key])
                        boxes_per_nifti[index] = boxes[scores > self.lesionThrs]
                        scores_per_nifti[index] = scores[scores > self.lesionThrs]
                        del inputDet, outputDet
                    except Exception as e:
                        print(f"Failed processing: {nifti_file} -> {str(e)}")
                        logging.info(f"Failed processing: {nifti_file} -> {str(e)}")
            if torch.cuda.is_available():
                torch.cuda.empty_cache()

        # classify bounding boxes (false-positive reduction), delegated to the
        # classification module.
        if self.use_classifier:
            classifications_per_nifti = self.classifier.Classify(niftis, boxes_per_nifti)
        else:
            classifications_per_nifti = [[] for _ in niftis]

        # assemble detections per scan
        results = {}
        for index, nifti_file in enumerate(niftis):
            boxes = boxes_per_nifti[index]
            scores = scores_per_nifti[index]
            if self.use_classifier:
                classifications = np.array(classifications_per_nifti[index]).flatten()
            else:
                classifications = np.ones(len(boxes), dtype=np.float32)  # no FP reduction
            shape = nib.load(nifti_file).shape
            # Build box-format detections first (overlap is defined on boxes).
            box_dets = []
            for idx, box in enumerate(boxes):
                score = float(scores[idx])
                classification = float(classifications[idx]) if idx < len(classifications) else 1.0
                if classification < self.noduleThrs or score < self.lesionThrs:
                    continue
                x_min, x_max, y_min, y_max, z_min, z_max = self._box_to_voxel_coords(box, shape)
                box_dets.append({
                    "nodule_score": score, "lesion_score": classification,
                    "x_min": x_min, "x_max": x_max,
                    "y_min": y_min, "y_max": y_max,
                    "z_min": z_min, "z_max": z_max,
                })
            # Ensemble FP reduction: merge spatially overlapping candidates.
            if self.ensemble.enabled:
                box_dets = self.ensemble.reduce(box_dets)
            # Format the output coordinates.
            results[nifti_file] = [self._format_detection(d) for d in box_dets]
        print("Detection done!")
        logging.info("Detection done!")
        return results
