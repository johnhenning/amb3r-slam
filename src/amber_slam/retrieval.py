"""ORB candidate retrieval. This is an explicit DBoW2 substitute."""

import cv2
import numpy as np


class LoopRetriever:
    def __init__(self, exclusion=6, min_score=0.12, vocabulary=None):
        self.orb = cv2.ORB_create(nfeatures=500)
        self.entries = []
        self.exclusion = exclusion
        self.min_score = min_score
        self.vocabulary = np.load(vocabulary, allow_pickle=False) if vocabulary else None
        self.matcher = cv2.BFMatcher(cv2.NORM_HAMMING)

    def descriptor(self, image):
        _, desc = self.orb.detectAndCompute(cv2.cvtColor(image, cv2.COLOR_RGB2GRAY), None)
        return desc

    def _histogram(self, desc):
        if desc is None:
            return np.zeros(len(self.vocabulary))
        # Nearest binary centroid; vocabulary produced by build_vocabulary.
        distances = np.unpackbits(
            np.bitwise_xor(desc[:, None], self.vocabulary[None]), axis=-1
        ).sum(-1)
        hist = np.bincount(distances.argmin(1), minlength=len(self.vocabulary)).astype(float)
        return hist / max(np.linalg.norm(hist), 1e-9)

    def query_and_add(self, image, top_k=3):
        desc = self.descriptor(image)
        representation = self._histogram(desc) if self.vocabulary is not None else desc
        candidates = []
        for i, previous in enumerate(self.entries):
            if len(self.entries) - i < self.exclusion:
                continue
            if self.vocabulary is not None:
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


def build_vocabulary(images, path, words=256, seed=0):
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
        assignments = np.array(assignments)
        for i in range(words):
            selected = d[assignments == i]
            if len(selected):
                centers[i] = np.packbits(np.unpackbits(selected, axis=-1).mean(0) >= 0.5)
    np.save(path, centers)
