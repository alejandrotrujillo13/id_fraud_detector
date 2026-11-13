"""Repeated faces: the same person on the front of more than one ID.

A forger can use their own face on several IDs with different numbers. If one of those IDs is in
the training folds and another in the test fold, the front classifier could recognise the face
instead of the forgery, and the cross-validation would look better than it will be with new
people. This module measures that: it reads the face of every front photo with two public
OpenCV models (YuNet to find the face, SFace to describe it with 128 numbers), compares every
pair of people and groups the ones whose faces match.

It is a check on the evaluation, not a part of the risk indicator. Everything runs on this
machine; the face descriptions are biometric data and stay under data/.

The two model files are public (github.com/opencv/opencv_zoo) and are kept in data/face_models/.
"""
from pathlib import Path

import cv2
import numpy as np

DETECTOR_FILE = "face_detection_yunet_2023mar.onnx"
RECOGNIZER_FILE = "face_recognition_sface_2021dec.onnx"
SFACE_THRESHOLD = 0.363  # cosine similarity above which SFace's authors call two faces the same person
CROP_SIZE = 112
ROTATIONS = (None, cv2.ROTATE_180, cv2.ROTATE_90_CLOCKWISE, cv2.ROTATE_90_COUNTERCLOCKWISE)


class FaceReader:
    """Finds the largest face of a photo and describes it. One reader per thread."""

    def __init__(self, model_dir: Path, max_side: int = 1280, min_score: float = 0.7):
        detector, recognizer = Path(model_dir) / DETECTOR_FILE, Path(model_dir) / RECOGNIZER_FILE
        for path in (detector, recognizer):
            if not path.exists():
                raise FileNotFoundError(f"{path} is missing: download it from github.com/opencv/opencv_zoo (see README)")
        self.max_side = max_side
        self.detector = cv2.FaceDetectorYN.create(str(detector), "", (320, 320), min_score, 0.3, 50)
        self.recognizer = cv2.FaceRecognizerSF.create(str(recognizer), "")

    def read(self, image: np.ndarray) -> dict | None:
        """`image` is BGR uint8. Returns the embedding (128 floats, length 1), the aligned face
        (112 x 112 BGR), the width of the face in pixels of the photo and the detector's score;
        None when no face is found. A photo taken sideways or upside down is tried rotated."""
        for rotation in ROTATIONS:
            turned = image if rotation is None else cv2.rotate(image, rotation)
            height, width = turned.shape[:2]
            scale = min(1.0, self.max_side / max(height, width))
            small = turned if scale == 1.0 else cv2.resize(turned, (round(width * scale), round(height * scale)),
                                                           interpolation=cv2.INTER_AREA)
            self.detector.setInputSize((small.shape[1], small.shape[0]))
            _, found = self.detector.detect(small)
            if found is None or len(found) == 0:
                continue
            face = found[np.argmax(found[:, 2] * found[:, 3])].copy()  # the portrait, not the ghost image
            face[:14] /= scale
            crop = self.recognizer.alignCrop(turned, face)
            feature = self.recognizer.feature(crop).ravel().astype(np.float32)
            norm = float(np.linalg.norm(feature))
            if norm == 0:
                continue
            return {"embedding": feature / norm, "crop": crop, "width": float(face[2]), "score": float(face[14])}
        return None


def similarities(embeddings: np.ndarray) -> np.ndarray:
    """Cosine similarity between every pair of rows (rows have length 1)."""
    return embeddings @ embeddings.T


def chance_rate(embeddings: np.ndarray, threshold: float) -> float:
    """Share of the pairs of `embeddings` whose similarity is above `threshold`. With people who
    are known to be different, it is how often two strangers are taken for the same person."""
    n = len(embeddings)
    if n < 2:
        return float("nan")
    upper = similarities(embeddings)[np.triu_indices(n, k=1)]
    return float((upper > threshold).mean())


def chance_threshold(embeddings: np.ndarray, rate: float) -> float:
    """The similarity that only a share `rate` of the pairs of `embeddings` exceeds."""
    n = len(embeddings)
    upper = similarities(embeddings)[np.triu_indices(n, k=1)]
    return float(np.quantile(upper, 1 - rate))


def pairs_above(embeddings: np.ndarray, threshold: float) -> np.ndarray:
    """Pairs (i, j) with i < j whose similarity is above the threshold, as an N x 2 array."""
    above = np.triu(similarities(embeddings) > threshold, k=1)
    return np.argwhere(above)


def face_groups(n: int, pairs: np.ndarray) -> np.ndarray:
    """Group number of each of `n` people: two people share a group when a chain of matching
    pairs joins them. People without a match are alone in their group."""
    parent = np.arange(n)

    def root(i: int) -> int:
        while parent[i] != i:
            parent[i] = parent[parent[i]]
            i = parent[i]
        return i

    for i, j in pairs:
        a, b = root(int(i)), root(int(j))
        if a != b:
            parent[max(a, b)] = min(a, b)
    return np.array([root(i) for i in range(n)])
