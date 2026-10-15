import os
import json
import shutil
import logging
import tempfile
import subprocess
from pathlib import Path

import nibabel as nib

## LESIONLOCATOR
## https://github.com/MIC-DKFZ/LesionLocator
## Zero-shot prompt-based (box or point) 3D lesion segmentation.
## Prompts are derived from annotation masks via the LesionLocator CLI, then
## LesionLocator_segment runs the nnU-Net-based model on the CT volume.


class LesionLocator(object):
    def __init__(self, cfg, logFile=None, checkIfExists=False):
        self.cfg = cfg
        self.checkIfExists = checkIfExists
        self.logFile = logFile
        llCfg = cfg.get('lesionlocator', {})
        self.checkpoint = Path(llCfg.get('checkpoint', os.path.join(
            "network_weights", "LesionLocatorCheckpoint")))
        self.LungTag = llCfg.get('lung_tag', None)
        self.NoduleTag = llCfg.get('nodule_tag', 'LL')
        self.device = llCfg.get('device', 'cuda')

    def GetPromptFromAnnotations(self, annFiles, annDir, output_dir, modality="box"):
        # Build a merged prompt.json (box or point) from one or more annotation
        # NIfTI masks using the LesionLocator CLI.
        annDir = Path(annDir)
        output_dir = Path(output_dir)
        # clear output_dir to avoid stale prompts
        if output_dir.is_dir():
            shutil.rmtree(output_dir)
        # copy annotations into a separate folder for the CLI to consume
        os.makedirs(annDir, exist_ok=True)
        for annFile in annFiles:
            shutil.copy(annFile, annDir / Path(annFile).name)
        # get .json file with annotations:
        cmd = ["LesionLocator_create_prompt_json",
               "-i", str(annDir),
               "-o", str(output_dir),
               "-label_type", "semantic"]
        _ = subprocess.run(cmd, check=True, capture_output=True, text=True)
        if modality == "box":
            prompts = list(Path(output_dir, "boxes").glob("*.json"))
        elif modality == "point":
            prompts = list(Path(output_dir, "points").glob("*.json"))
        else:
            raise ValueError(f"Unsupported modality '{modality}'. Use 'box' or 'point'.")
        merged = {}
        counter = 1
        for p in prompts:
            with open(p) as f:
                data = json.load(f)
            for v in data.values():
                merged[str(counter)] = v
                counter += 1
        merged_path = Path(output_dir, "prompt.json")
        with open(merged_path, "w") as f:
            json.dump(merged, f)
        return merged_path

    def SegmentLesions(self, niftis, annFiles, outputFolder, modality="box",
                       getLesionsBB=False):
        # Segment lesions in each CT volume, prompted by its annotation mask.
        # Produces one merged mask per scan: {stem}_{NoduleTag}.nii.gz.
        outputFolder = Path(outputFolder)
        outputFolder.mkdir(parents=True, exist_ok=True)
        for nifti, annFile in zip(niftis, annFiles):
            if not os.path.exists(annFile):
                logging.warning(f"Annotation not found for {nifti}, skipping.")
                continue
            outputFilename = self.GetOutputFilename(nifti, outputFolder, tag=self.NoduleTag)
            if self.checkIfExists and os.path.exists(outputFilename):
                continue
            # Isolate each scan's intermediate files in a temp job dir.
            jobDir = Path(tempfile.mkdtemp(prefix="ll_job_", dir=outputFolder))
            annDir = jobDir / "ann"
            job_output_dir = jobDir / "nodule_segmentation"
            try:
                prompt = self.GetPromptFromAnnotations(
                    [annFile], annDir, job_output_dir, modality=modality)
                with open(prompt) as f:
                    prompt_data = json.load(f)
                if not prompt_data:
                    print(f"Skipping empty prompt: {prompt}")
                    logging.warning(f"Empty prompt for {nifti}, skipping.")
                    continue
                cmd = ["LesionLocator_segment",
                       "-i", str(nifti),
                       "-p", str(prompt),
                       "-t", modality,
                       "-o", str(job_output_dir),
                       "-m", str(self.checkpoint),
                       "-device", self.device]
                _ = subprocess.run(cmd, check=True, capture_output=True, text=True)
                segFiles = sorted(str(pp) for pp in job_output_dir.glob("*.nii.gz"))
                if not segFiles:
                    raise RuntimeError("No segmentation file produced.")
                self.MergeLesions(segFiles, outputFilename, getLesionsBB=getLesionsBB)
            except subprocess.CalledProcessError as e:
                print(f"LesionLocator_segment FAILED for: {nifti}")
                print(f"STDOUT: {e.stdout}")
                print(f"STDERR: {e.stderr}")
                logging.error(f"LesionLocator segmentation failed for {nifti}")
            finally:
                shutil.rmtree(jobDir, ignore_errors=True)
        print("LesionLocator segmentation completed!")
        logging.info("LesionLocator segmentation completed!")

    def MergeLesions(self, segFiles, outputFilename, getLesionsBB=False):
        # Merge per-lesion instance maps into a single binary mask.
        if len(segFiles) == 0:
            return
        merged_data = None
        ref_img = None
        ref_header = None
        for idx, segFile in enumerate(segFiles):
            img = nib.load(segFile)
            data = img.get_fdata()
            if idx == 0:
                merged_data = (data > 0)
                ref_img = img
                ref_header = img.header.copy()
            else:
                merged_data[data > 0] = True
        nifti_img = nib.Nifti1Image(
            merged_data.astype("uint8"), affine=ref_img.affine, header=ref_header)
        nib.save(nifti_img, str(outputFilename))
        logging.info(f"Merged lesion segmentation saved as: {outputFilename}")
        print(f"Merged lesion segmentation saved as: {outputFilename}")

    def GetOutputFilename(self, nifti, outputFolder, tag=""):
        nifti_path = Path(nifti)
        outputFolder = Path(outputFolder)
        outputFolder.mkdir(parents=True, exist_ok=True)
        if nifti_path.name.endswith(".nii.gz"):
            stem = nifti_path.name[:-7]  # remove ".nii.gz"
            filename = f"{stem}_{tag}.nii.gz" if tag else f"{stem}.nii.gz"
        else:
            filename = f"{nifti_path.stem}_{tag}{nifti_path.suffix}" if tag \
                    else f"{nifti_path.name}"
        return outputFolder / filename
