"""Segmenter module.

A thin, config-driven dispatcher over the available segmentation backends.
The user selects a backend by name (``segmenter.model`` in the config) and the
Segmenter instantiates it and exposes a uniform ``Segment(...)`` interface.

Adding a new backend is a two-step change:
  1. implement it under ``src/baselines/`` (or wherever it lives), and
  2. register a builder here with ``@register_segmenter("<name>")`` that returns
     ``(backend, segment_fn)`` where ``segment_fn`` adapts the backend's own API
     to the common ``Segment(niftis, annFiles, outputFolder, **kwargs)`` call.

Only LesionLocator is wired up so far.
"""
import logging

from src.baselines.lesionlocator import LesionLocator


# name -> builder(cfg, logFile, checkIfExists) -> (backend, segment_fn)
SEGMENTER_REGISTRY = {}


def register_segmenter(name):
    def deco(builder):
        SEGMENTER_REGISTRY[name] = builder
        return builder
    return deco


@register_segmenter("lesionlocator")
def _build_lesionlocator(cfg, logFile, checkIfExists):
    backend = LesionLocator(cfg, logFile=logFile, checkIfExists=checkIfExists)

    def segment(niftis, annFiles, outputFolder, **kwargs):
        # LesionLocator is prompted by annotation masks (box/point).
        return backend.SegmentLesions(niftis, annFiles, outputFolder, **kwargs)

    return backend, segment


# Additional backends register here as they are wired in, e.g.:
#
#   @register_segmenter("medsam2")
#   def _build_medsam2(cfg, logFile, checkIfExists):
#       backend = MedSAM2(cfg, logFile=logFile, checkIfExists=checkIfExists)
#       def segment(niftis, annFiles, outputFolder, **kwargs):
#           prompts = backend.ExtractPromptsFromMask(niftis, annFiles)
#           return backend.SegmentNodules(niftis, prompts, outputFolder)
#       return backend, segment


class Segmenter(object):
    def __init__(self, cfg, logFile=None, checkIfExists=False):
        self.cfg = cfg
        self.logFile = logFile
        self.name = cfg.get('segmenter', {}).get('model', 'lesionlocator')
        if self.name not in SEGMENTER_REGISTRY:
            available = ", ".join(sorted(SEGMENTER_REGISTRY)) or "(none)"
            raise ValueError(
                f"Unknown segmenter '{self.name}'. Available: {available}.")
        logging.info(f"Segmenter backend: {self.name}")
        print(f"Segmenter backend: {self.name}")
        self.backend, self._segment = SEGMENTER_REGISTRY[self.name](
            cfg, logFile, checkIfExists)

    def Segment(self, niftis, annFiles, outputFolder, **kwargs):
        """Segment lesions in each CT volume, prompted by its annotation mask.

        Delegates to the selected backend. Returns whatever the backend returns
        (LesionLocator writes merged masks to `outputFolder`).
        """
        return self._segment(niftis, annFiles, outputFolder, **kwargs)

    @staticmethod
    def available_backends():
        return sorted(SEGMENTER_REGISTRY)
