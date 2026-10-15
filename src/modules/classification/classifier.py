import os
import logging

import numpy as np
import torch
import nibabel as nib
import monai
from monai.data import DataLoader, Dataset
from monai.transforms import ScaleIntensityRanged

from src.utils.generate_transforms import generate_classification_test_transform

## CLASSIFICATION MODULE
## ResNet50 (MONAI, 3D) false-positive-reduction classifier.
## Given candidate boxes from the Detector, it crops a fixed-size patch around
## each box and predicts the probability that it contains a true lesion.

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")


class BBClassifierDataset(Dataset):
    """Crops fixed-size patches around detected boxes for the FP-reduction classifier."""

    def __init__(self, nifti_paths, boxes_per_nifti, classification_patch_size=None):
        self.nifti_paths = nifti_paths
        self.boxes_per_nifti = boxes_per_nifti
        self.classification_patch_size = classification_patch_size or [64, 64, 64]
        self.transform = self.DefineClassificationDataTransforms()
        self.samples = []
        self.dict_path_patient = {}
        for path, boxes in zip(nifti_paths, boxes_per_nifti):
            for box in boxes:
                self.samples.append((path, box))
        print("Total samples for classification:", len(self.samples))

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, idx):
        path, box = self.samples[idx]

        if path not in self.dict_path_patient:
            self.dict_path_patient = {}
            self.dict_path_patient[path] = nib.load(path).get_fdata()
        img = self.dict_path_patient[path]

        # Crop to bounding box
        z_min_f, x_min_f, y_min_f, z_max_f, x_max_f, y_max_f = box
        # Compute the center of the bounding box
        x_center = int(np.round((x_min_f + x_max_f) / 2))
        y_center = int(np.round((y_min_f + y_max_f) / 2))
        z_center = int(np.round((z_min_f + z_max_f) / 2))
        half = [s // 2 for s in self.classification_patch_size]
        x_max = min(img.shape[0], img.shape[0] - 1 - (x_center - half[0]))
        x_min = max(0, img.shape[0] - 1 - (x_center + half[0]))
        y_min = max(0, y_center - half[1])
        y_max = min(img.shape[1], y_center + half[1])
        z_min = max(0, z_center - half[2])
        z_max = min(img.shape[2], z_center + half[2])
        crop = img[x_min:x_max, y_min:y_max, z_min:z_max]
        # pad if necessary
        pad_x_before = max(0, -x_min)
        pad_x_after = max(0, x_max - img.shape[0])
        pad_y_before = max(0, -y_min)
        pad_y_after = max(0, y_max - img.shape[1])
        pad_z_before = max(0, -z_min)
        pad_z_after = max(0, z_max - img.shape[2])
        crop = np.pad(crop,
                      ((pad_x_before, pad_x_after),
                       (pad_y_before, pad_y_after),
                       (pad_z_before, pad_z_after)),
                      mode='constant',
                      constant_values=0)
        if self.transform:
            out = self.transform({"image": crop})
        output = {'image': out['image'], 'path': path}
        return output

    def DefineClassificationDataTransforms(self):
        intensity_transform = ScaleIntensityRanged(
            keys=["image"],
            a_min=-1024,
            a_max=800.0,
            b_min=0.0,
            b_max=1.0,
            clip=True,
        )
        test_transforms = generate_classification_test_transform(
            image_key="image",
            img_patch_size=self.classification_patch_size,
            intensity_transform=intensity_transform,
        )
        return test_transforms


class Classifier(object):
    def __init__(self, cfg, logFile=None):
        self.cfg = cfg
        self.logFile = logFile
        clf = cfg.get('classifier', {})
        self.name = clf.get('name', 'resnet50WSPP')
        self.weights = clf.get('weights', 'resnet50WSPP.pt')
        self.threshold = clf.get('threshold', 0.0)
        self.patch_size = clf.get('patch_size', [64, 64, 64])
        self.weights_dir = cfg.get('network_weights_dir', 'network_weights')
        self.model = self.GetClassifier()

    def GetClassifier(self):
        logging.info("Loading ResNet50 for lesion classification...")
        print("Loading ResNet50 for lesion classification...")
        modelPath = os.path.join(self.weights_dir, self.name, self.weights)
        model = monai.networks.nets.resnet50(pretrained=False,
                                             spatial_dims=3,
                                             n_input_channels=1,
                                             num_classes=2)
        ckpt = torch.load(modelPath, map_location="cpu")
        model.load_state_dict(ckpt, strict=True)
        classifier = model.to(device)
        classifier.eval()
        logging.info("ResNet50 loaded.")
        print("ResNet50 loaded.")
        return classifier

    def Classify(self, niftis, boxes_per_nifti, batch_size=100):
        """Score each candidate box.

        Returns a list aligned with `niftis`; each entry is a list of lesion
        probabilities (in [0, 1]) for that scan's boxes, in box order.
        """
        niftis = [os.path.abspath(str(n)) for n in niftis]
        nifti_to_idx = {n: i for i, n in enumerate(niftis)}
        dataset = BBClassifierDataset(niftis, boxes_per_nifti,
                                      classification_patch_size=self.patch_size)
        loader = DataLoader(dataset,
                            batch_size=batch_size,
                            num_workers=4,
                            pin_memory=torch.cuda.is_available())
        classifications_per_nifti = [[] for _ in niftis]
        softmax = torch.nn.Softmax(dim=1)
        with torch.no_grad():
            for class_data in loader:
                nifti_paths = [os.path.abspath(p) for p in class_data["path"]]
                indexes = [nifti_to_idx[p] for p in nifti_paths]
                inputClass = class_data["image"].to(device)
                outputClass = self.model(inputClass)
                softmaxOutput = softmax(outputClass)
                classifications = softmaxOutput[:, 1].cpu().numpy()
                for idx, cc in zip(indexes, classifications):
                    classifications_per_nifti[idx].append(float(cc))
                del inputClass, outputClass
            if torch.cuda.is_available():
                torch.cuda.empty_cache()
        return classifications_per_nifti
