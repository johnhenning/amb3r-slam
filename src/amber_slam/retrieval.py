"""ORB candidate retrieval. This is an explicit DBoW2 substitute."""

from __future__ import annotations

from collections.abc import Iterable
from typing import Protocol

import cv2
import numpy as np

from .contracts import Array, PathLike


class CandidateRetriever(Protocol):
    """Candidate retrieval boundary; geometric verification belongs to the backend."""

    def query_and_add(self, image: Array, top_k: int = 3) -> list[int]: ...


class LoopRetriever:
    def __init__(
        self, exclusion: int = 6, min_score: float = 0.12, vocabulary: PathLike | None = None
    ) -> None:
        self.orb = cv2.ORB.create(nfeatures=500)
        self.entries: list[Array | None] = []
        self.exclusion = exclusion
        self.min_score = min_score
        self.vocabulary = np.load(vocabulary, allow_pickle=False) if vocabulary else None
        self.matcher = cv2.BFMatcher(cv2.NORM_HAMMING)

    def descriptor(self, image: Array) -> Array | None:
        _, desc = self.orb.detectAndCompute(
            cv2.cvtColor(image, cv2.COLOR_RGB2GRAY), np.empty(0, dtype=np.uint8)
        )
        return desc

    def _histogram(self, desc: Array | None) -> Array:
        assert self.vocabulary is not None
        if desc is None:
            return np.zeros(len(self.vocabulary))
        # Nearest binary centroid; vocabulary produced by build_vocabulary.
        distances = np.unpackbits(
            np.bitwise_xor(desc[:, None], self.vocabulary[None]), axis=-1
        ).sum(-1)
        hist = np.bincount(distances.argmin(1), minlength=len(self.vocabulary)).astype(float)
        return hist / max(float(np.linalg.norm(hist)), 1e-9)

    def query_and_add(self, image: Array, top_k: int = 3) -> list[int]:
        desc = self.descriptor(image)
        representation = self._histogram(desc) if self.vocabulary is not None else desc
        candidates = []
        for i, previous in enumerate(self.entries):
            if len(self.entries) - i < self.exclusion:
                continue
            if self.vocabulary is not None:
                assert representation is not None and previous is not None
                score = float(representation @ previous)
            elif desc is None or previous is None or len(previous) < 2:
                continue
            else:
                matches = self.matcher.knnMatch(desc, previous, k=2)
                good = sum(len(m) == 2 and m[0].distance < 0.75 * m[1].distance for m in matches)
                score = good / max(len(desc), len(previous))
            if score >= self.min_score:
                candidates.append((score, i))
        self.entries.append(representation)
        return [i for _, i in sorted(candidates, reverse=True)[:top_k]]


def build_vocabulary(
    images: Iterable[Array], path: PathLike, words: int = 256, seed: int = 0
) -> None:
    """Binary majority centroids initialized from training ORB descriptors."""
    retrieval = LoopRetriever()
    arrays = [retrieval.descriptor(x) for x in images]
    arrays = [a for a in arrays if a is not None]
    if not arrays:
        raise ValueError("No ORB features in vocabulary training images")
    d = np.concatenate(arrays)
    rng = np.random.default_rng(seed)
    if len(d) < words:
        raise ValueError("Not enough descriptors for requested vocabulary")
    d = d[rng.choice(len(d), min(len(d), 10000), replace=False)]
    centers = d[rng.choice(len(d), words, replace=False)]
    for _ in range(10):
        assignments = []
        for chunk in np.array_split(d, max(1, len(d) // 128)):
            dist = np.unpackbits(np.bitwise_xor(chunk[:, None], centers[None]), axis=-1).sum(-1)
            assignments.extend(dist.argmin(1))
        assignment_ids = np.array(assignments)
        for i in range(words):
            selected = d[assignment_ids == i]
            if len(selected):
                centers[i] = np.packbits(np.unpackbits(selected, axis=-1).mean(0) >= 0.5)
    np.save(path, centers)
