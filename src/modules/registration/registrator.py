"""Registration module (uniGradICON / GradICON).

Deformably registers a moving CT to a fixed CT with the pretrained uniGradICON
network (a GradICON model with universal weights), then warps the moving image
-- and, optionally, its masks/annotations -- onto the fixed image's grid.

https://github.com/uncbiag/uniGradICON

The heavy Dataset coupling of the source implementation is dropped in favour of
uniGradICON's ITK-native API (`preprocess` + `itk_wrapper.register_pair` +
`itk.resample_image_filter`), which handles image geometry directly.
"""
import os
import logging
from pathlib import Path
from contextlib import contextmanager

import numpy as np
import torch
import itk
from skimage.metrics import structural_similarity as ssim

from unigradicon import get_unigradicon, preprocess
from icon_registration import itk_wrapper

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")


@contextmanager
def _chdir(path):
    prev = os.getcwd()
    os.chdir(path)
    try:
        yield
    finally:
        os.chdir(prev)


class Registrator(object):
    def __init__(self, cfg, logFile=None):
        self.cfg = cfg
        self.logFile = logFile
        reg = cfg.get('registration', {})
        self.fixed_time = reg.get('fixed_time', 'T0')  # reference timepoint (moving scans register onto it)
        self.modality = reg.get('modality', 'ct')
        self.finetune_steps = reg.get('finetune_steps', None)
        self.register_with_mask = reg.get('register_with_mask', False)
        # get_unigradicon() reads weights from a cwd-relative path
        # ("network_weights/unigradicon1.0/..."); resolve it against the repo so
        # the pretrained weights are found regardless of the caller's cwd.
        self.weights_root = os.path.abspath(
            cfg.get('weights_root', Path(__file__).resolve().parents[3]))
        self.net = self.getUnigradicon()

    def getUnigradicon(self):
        logging.info("Loading UniGradICON network for registration...")
        print("Loading UniGradICON network for registration...")
        with _chdir(self.weights_root):
            net = get_unigradicon(apply_intensity_conservation_loss=True)
        logging.info("UniGradICON network loaded.")
        print("UniGradICON network loaded.")
        return net.to(device).eval()

    # ------------------------------------------------------------------ #
    def Register(self, fixedNifti, movingNifti, outputFolder,
                 movingMasks=None, fixedMask=None, movingMask=None):
        """Register `movingNifti` onto `fixedNifti` and warp it to the fixed grid.

        Parameters
        ----------
        movingMasks : list of (path, tag) or None
            Additional moving-space label maps to warp with the same transform
            (nearest-neighbour). Each is saved as {moving_stem}_{tag}_reg.nii.gz.
        fixedMask, movingMask : path or None
            Fixed / moving lesion masks. With register_with_mask=True the
            registration ignores these masked-out (lesion) regions when
            estimating the field. These are the exclusion masks, distinct from
            `movingMasks` (which are merely warped along with the image).

        Returns
        -------
        dict with 'warped_image', warped mask paths, and before/after metrics.
        """
        result = self.RegisterOrLoad(fixedNifti, movingNifti, outputFolder,
                                     movingMasks=movingMasks, fixedMask=fixedMask,
                                     movingMask=movingMask, checkIfExists=False)
        # Metrics before (moving vs fixed) and after (warped vs fixed).
        result['metrics_before'] = self.ComputeMetrics(fixedNifti, movingNifti)
        result['metrics_after'] = self.ComputeMetrics(fixedNifti, result['warped_image'])
        logging.info(f"Registered {movingNifti} -> {fixedNifti}: "
                     f"SSIM {result['metrics_before'].get('ssim')} -> "
                     f"{result['metrics_after'].get('ssim')}")
        return result

    def _compute_transform(self, fixedNifti, movingNifti, fixed_itk, moving_itk,
                           movingMask=None, fixedMask=None):
        """Estimate (phi_AB, phi_BA): moving->fixed and its inverse (fixed->moving)."""
        fixed_pre = preprocess(fixed_itk, self.modality)
        moving_pre = preprocess(moving_itk, self.modality)
        if self.register_with_mask and fixedMask is None:
            msg = ("register_with_mask=True but no fixedMask was provided; "
                   "falling back to whole-image registration (lesion regions NOT ignored).")
            logging.warning(msg)
            print(f"WARNING: {msg}")
        if self.register_with_mask and fixedMask is not None:
            # Exclude masked (lesion) regions from the similarity estimation.
            fixed_seg = self._read_inv_mask(fixedMask, fixed_itk)
            moving_seg = (self._read_inv_mask(movingMask, moving_itk)
                          if movingMask is not None else None)
            fixed_pre = preprocess(fixed_itk, self.modality, segmentation=fixed_seg)
            moving_pre = preprocess(moving_itk, self.modality, segmentation=moving_seg)
        phi_AB, phi_BA = itk_wrapper.register_pair(
            self.net, moving_pre, fixed_pre, finetune_steps=self.finetune_steps)
        return phi_AB, phi_BA

    def RegisterOrLoad(self, fixedNifti, movingNifti, outputFolder,
                       movingMasks=None, fixedMask=None, movingMask=None,
                       checkIfExists=True):
        """Register moving -> fixed, reusing a cached transform/warp if present.

        When `checkIfExists` and both the warped image and the saved displacement
        field already exist, the (expensive) network registration is skipped and
        the cached field is reloaded to (re)warp the moving masks. Otherwise a
        fresh registration is run and the field is saved for next time.

        Returns dict with 'warped_image', 'warped_masks', 'field' and 'reused'.
        """
        outputFolder = Path(outputFolder)
        outputFolder.mkdir(parents=True, exist_ok=True)
        moving_stem = Path(movingNifti).name[:-7]
        # Mask-aware cache: masked and whole-image registrations are cached under
        # distinct names, so toggling register_with_mask never reuses a stale
        # field/warp from the other mode.
        mtag = "_masked" if (self.register_with_mask and fixedMask is not None) else ""
        warped_path = outputFolder / f"{moving_stem}{mtag}_reg.nii.gz"
        field_path = outputFolder / f"{moving_stem}{mtag}_phi_AB.h5"       # moving -> fixed
        field_inv_path = outputFolder / f"{moving_stem}{mtag}_phi_BA.h5"   # fixed  -> moving

        fixed_itk = itk.imread(str(fixedNifti), itk.F)
        moving_itk = itk.imread(str(movingNifti), itk.F)

        phi_AB, reused = None, False
        if (checkIfExists and warped_path.exists()
                and field_path.exists() and field_inv_path.exists()):
            try:
                phi_AB = itk.transformread(str(field_path))[0]
                reused = True
                print(f"[registration] reusing cached registration: {warped_path.name}")
                logging.info(f"Reusing cached registration for {movingNifti}")
            except Exception as e:
                logging.warning(f"Could not reload cached field ({e}); re-registering.")
                phi_AB = None

        # The registration fields are always cached to disk: they let a repeated
        # run reuse the transform (register-if-not-present) and are reused to
        # warp candidates, GT and outputs (phi_AB: moving->fixed, phi_BA: the
        # inverse, fixed->moving).
        if phi_AB is None:
            phi_AB, phi_BA = self._compute_transform(
                fixedNifti, movingNifti, fixed_itk, moving_itk, movingMask, fixedMask)
            self._warp(moving_itk, fixed_itk, phi_AB, warped_path, nearest=False)
            itk.transformwrite([phi_AB], str(field_path))
            itk.transformwrite([phi_BA], str(field_inv_path))

        result = {'warped_image': str(warped_path), 'reused': reused,
                  'field': str(field_path),          # phi_AB (moving -> fixed)
                  'field_inv': str(field_inv_path)}   # phi_BA (fixed -> moving)

        # Warp moving-space masks/annotations (always; they may have changed).
        warped_masks = {}
        for maskPath, tag in (movingMasks or []):
            mask_itk = itk.imread(str(maskPath), itk.F)
            wpath = outputFolder / f"{moving_stem}{mtag}_{tag}_reg.nii.gz"
            self._warp(mask_itk, fixed_itk, phi_AB, wpath, nearest=True)
            warped_masks[tag] = str(wpath)
        if warped_masks:
            result['warped_masks'] = warped_masks
        return result

    def WarpBack(self, fixedFrameNifti, movingRefNifti, field_inv_path, out_path,
                 nearest=True):
        """Warp a fixed-frame image back onto the moving image's grid via phi_BA.

        Used to bring a segmentation produced in the fixed (T0) frame back into
        the follow-up's own space for own-space evaluation.
        """
        phi_BA = itk.transformread(str(field_inv_path))[0]
        moved_itk = itk.imread(str(fixedFrameNifti), itk.F)
        ref_itk = itk.imread(str(movingRefNifti), itk.F)
        return self._warp(moved_itk, ref_itk, phi_BA, out_path, nearest=nearest)

    def RegisterSeries(self, fixedNifti, movingNiftis, outputFolder,
                       movingMasksPerScan=None):
        """Register each moving scan onto the fixed scan. Returns list of dicts."""
        results = []
        for i, moving in enumerate(movingNiftis):
            masks = movingMasksPerScan[i] if movingMasksPerScan else None
            results.append(self.Register(fixedNifti, moving, outputFolder,
                                         movingMasks=masks))
        return results

    # ------------------------------------------------------------------ #
    @staticmethod
    def _warp(moving_itk, fixed_itk, phi_AB, out_path, nearest=False):
        moving_itk, maybe_cast_back = Registrator.maybe_cast(moving_itk)
        if nearest:
            interpolator = itk.NearestNeighborInterpolateImageFunction.New(moving_itk)
        else:
            interpolator = itk.LinearInterpolateImageFunction.New(moving_itk)
        warped = itk.resample_image_filter(
            moving_itk,
            transform=phi_AB,
            interpolator=interpolator,
            use_reference_image=True,
            reference_image=fixed_itk)
        warped = maybe_cast_back(warped)
        itk.imwrite(warped, str(out_path))
        return str(out_path)

    @staticmethod
    def _read_inv_mask(mask_path, ref_itk):
        """Read a mask and return an inverse (1 outside lesion) ITK float image."""
        arr = itk.array_from_image(itk.imread(str(mask_path), itk.F))
        inv = (arr <= 0).astype(np.float32)
        out = itk.image_from_array(inv)
        out.CopyInformation(ref_itk)
        return out

    @staticmethod
    def maybe_cast(img):
        """Cast to double for ITK interpolators, returning a cast-back fn (from unigradicon)."""
        maybe_cast_back = lambda x: x
        if str((type(img), itk.D)) not in itk.NearestNeighborInterpolateImageFunction.GetTypesAsList():
            if type(img) in (itk.Image[itk.ULL, 3], itk.Image[itk.UL, 3]):
                raise Exception("Label maps of type unsigned long may have unrepresentable values")
            maybe_cast_back = itk.CastImageFilter[itk.Image[itk.D, 3], type(img)].New()
            img = itk.CastImageFilter[type(img), itk.Image[itk.D, 3]].New()(img)
        return img, maybe_cast_back

    @staticmethod
    def ComputeMetrics(targetPath, sourcePath):
        """NCC, MSE, SSIM between two same-grid images (0 if shapes differ)."""
        target = itk.array_from_image(itk.imread(str(targetPath), itk.F)).astype(np.float32)
        source = itk.array_from_image(itk.imread(str(sourcePath), itk.F)).astype(np.float32)
        if target.shape != source.shape:
            logging.warning(f"Shapes differ: {target.shape} vs {source.shape}; skipping metrics.")
            return {'ncc': None, 'mse': None, 'ssim': None}
        ncc = float(np.corrcoef(target.flatten(), source.flatten())[0, 1])
        mse = float(np.mean((target - source) ** 2))
        rng = float(source.max() - source.min())
        ssim_val = float(ssim(target, source, data_range=rng)) if rng > 0 else 1.0
        return {'ncc': ncc, 'mse': mse, 'ssim': ssim_val}
