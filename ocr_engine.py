"""Open-source OCR backends for scanned / image-only PDF pages.

Backend priority:
  1. RapidOCR (``rapidocr-onnxruntime``) -- Apache-2.0, pip-only, ships PP-OCRv4
     ONNX models, needs no system binary. This is the default.
  2. pytesseract -- used when the Tesseract binary happens to be installed.

Both backends are normalised to the same output: a list of :class:`Fragment`,
i.e. text plus a bounding box in *image pixel* space. The caller rescales those
boxes into PDF-point space so that the OCR path and the digital-text path feed
the exact same table-reconstruction code.
"""

from __future__ import annotations

import logging
from dataclasses import dataclass
from typing import List, Optional

import numpy as np

logger = logging.getLogger(__name__)

__all__ = ["Fragment", "OCREngine", "deskew", "binarize"]


@dataclass
class Fragment:
    """A piece of text with its bounding box.

    Coordinates follow the PDF/image convention: origin top-left, ``top`` is
    smaller than ``bottom``.
    """

    text: str
    x0: float
    top: float
    x1: float
    bottom: float
    conf: float = 1.0

    @property
    def cx(self) -> float:
        return (self.x0 + self.x1) / 2.0

    @property
    def cy(self) -> float:
        return (self.top + self.bottom) / 2.0

    @property
    def width(self) -> float:
        return self.x1 - self.x0

    @property
    def height(self) -> float:
        return self.bottom - self.top

    def scaled(self, factor: float) -> "Fragment":
        return Fragment(
            self.text,
            self.x0 * factor,
            self.top * factor,
            self.x1 * factor,
            self.bottom * factor,
            self.conf,
        )


# ---------------------------------------------------------------------------
# Image helpers
# ---------------------------------------------------------------------------


def _to_gray(image: np.ndarray) -> np.ndarray:
    import cv2

    if image.ndim == 3:
        return cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    return image


def binarize(image: np.ndarray) -> np.ndarray:
    """Return a white-text-on-black mask, which is what morphology wants."""
    import cv2

    gray = _to_gray(image)
    # Otsu on a slightly blurred image copes with JPEG noise in scans.
    blurred = cv2.GaussianBlur(gray, (3, 3), 0)
    _, mask = cv2.threshold(blurred, 0, 255, cv2.THRESH_BINARY_INV | cv2.THRESH_OTSU)
    return mask


def deskew(image: np.ndarray, max_angle: float = 8.0) -> np.ndarray:
    """Rotate a scan so its text lines are horizontal.

    The angle comes from the dominant near-horizontal line found by a Hough
    transform, which is far more stable on tables than ``minAreaRect`` over all
    ink. Rotations beyond ``max_angle`` are ignored -- they are almost always a
    misdetection rather than a genuinely crooked scan.
    """
    import cv2

    mask = binarize(image)
    edges = cv2.Canny(mask, 50, 150, apertureSize=3)
    lines = cv2.HoughLinesP(
        edges, 1, np.pi / 720, threshold=120,
        minLineLength=max(60, image.shape[1] // 6), maxLineGap=12,
    )
    if lines is None:
        return image

    angles = []
    # OpenCV returns (N, 1, 4) on most builds but plain (N, 4) on some, so
    # normalise rather than trusting the shape.
    for x1, y1, x2, y2 in np.asarray(lines).reshape(-1, 4):
        angle = np.degrees(np.arctan2(y2 - y1, x2 - x1))
        if abs(angle) <= max_angle:
            angles.append(angle)
    if not angles:
        return image

    angle = float(np.median(angles))
    if abs(angle) < 0.15:  # already straight enough; skip the resample
        return image

    h, w = image.shape[:2]
    matrix = cv2.getRotationMatrix2D((w / 2, h / 2), angle, 1.0)
    return cv2.warpAffine(
        image, matrix, (w, h),
        flags=cv2.INTER_CUBIC, borderMode=cv2.BORDER_REPLICATE,
    )


# ---------------------------------------------------------------------------
# OCR engine
# ---------------------------------------------------------------------------


class OCREngine:
    """Lazily-loaded OCR wrapper.

    Model loading costs a few seconds, so it is deferred until a page actually
    turns out to be scanned. A PDF that is fully digital never pays for it.
    """

    def __init__(self, backend: Optional[str] = None):
        self._requested = backend
        self._impl = None
        self._name = "uninitialised"
        self._failed = False

    # -- backend discovery --------------------------------------------------

    def _load(self) -> None:
        if self._impl is not None or self._failed:
            return

        order = [self._requested] if self._requested else ["rapidocr", "tesseract"]
        errors = []
        for name in order:
            try:
                if name == "rapidocr":
                    self._impl = _RapidOCRBackend()
                elif name == "tesseract":
                    self._impl = _TesseractBackend()
                else:
                    raise ValueError(f"Unknown OCR backend {name!r}")
                self._name = name
                logger.info("OCR backend: %s", name)
                return
            except Exception as exc:  # noqa: BLE001 - report all, fail at the end
                errors.append(f"{name}: {exc}")

        self._failed = True
        self._name = "none"
        logger.warning("No OCR backend available (%s)", "; ".join(errors))

    @property
    def name(self) -> str:
        self._load()
        return self._name

    @property
    def available(self) -> bool:
        self._load()
        return self._impl is not None

    # -- public API ---------------------------------------------------------

    def read(self, image: np.ndarray) -> List[Fragment]:
        """OCR a full page image and return positioned text fragments."""
        self._load()
        if self._impl is None:
            return []
        try:
            return self._impl.read(image)
        except Exception as exc:  # noqa: BLE001 - a bad page must not kill the job
            logger.warning("OCR failed on page image: %s", exc)
            return []

    def read_text(self, image: np.ndarray) -> str:
        """OCR a small crop (a single table cell) and return plain text."""
        fragments = self.read(image)
        if not fragments:
            return ""
        fragments.sort(key=lambda f: (round(f.cy, 1), f.x0))
        return " ".join(f.text for f in fragments).strip()


class _RapidOCRBackend:
    """PP-OCRv4 via onnxruntime. Detection boxes are phrase-level."""

    def __init__(self):
        try:
            from rapidocr_onnxruntime import RapidOCR  # type: ignore
        except ImportError:
            from rapidocr import RapidOCR  # type: ignore  # v2+ package name
        self._engine = RapidOCR()

    def read(self, image: np.ndarray) -> List[Fragment]:
        raw = self._engine(image)

        # v1 returns (result, elapse); v2+ returns an object with .boxes/.txts.
        boxes = txts = scores = None
        if isinstance(raw, tuple):
            raw = raw[0]
        if raw is None:
            return []
        if hasattr(raw, "boxes"):
            boxes, txts, scores = raw.boxes, raw.txts, raw.scores
            if boxes is None:
                return []
            triples = zip(boxes, txts, scores)
        else:
            triples = ((item[0], item[1], item[2]) for item in raw)

        fragments = []
        for box, text, score in triples:
            text = (text or "").strip()
            if not text:
                continue
            pts = np.asarray(box, dtype=float).reshape(-1, 2)
            fragments.append(
                Fragment(
                    text=text,
                    x0=float(pts[:, 0].min()),
                    top=float(pts[:, 1].min()),
                    x1=float(pts[:, 0].max()),
                    bottom=float(pts[:, 1].max()),
                    conf=float(score),
                )
            )
        return fragments


class _TesseractBackend:
    """Tesseract via pytesseract. Detection boxes are word-level."""

    MIN_CONF = 30

    def __init__(self):
        import pytesseract  # type: ignore

        pytesseract.get_tesseract_version()  # raises if the binary is missing
        self._pytesseract = pytesseract

    def read(self, image: np.ndarray) -> List[Fragment]:
        data = self._pytesseract.image_to_data(
            image,
            config="--oem 3 --psm 6",
            output_type=self._pytesseract.Output.DICT,
        )

        fragments = []
        for i, text in enumerate(data["text"]):
            text = (text or "").strip()
            if not text:
                continue
            try:
                conf = float(data["conf"][i])
            except (TypeError, ValueError):
                conf = -1.0
            if conf < self.MIN_CONF:
                continue
            x, y = data["left"][i], data["top"][i]
            w, h = data["width"][i], data["height"][i]
            fragments.append(
                Fragment(text, float(x), float(y), float(x + w), float(y + h), conf / 100.0)
            )
        return fragments
