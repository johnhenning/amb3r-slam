"""DA3 adapter contract, using a fake public API prediction (not model accuracy)."""

from __future__ import annotations

from collections.abc import Sequence
from types import SimpleNamespace
from unittest.mock import patch

import numpy as np

from amber_slam.adapters import DA3Model
from amber_slam.contracts import Array
from amber_slam.types import Frame


def test_da3_w2c_conversion_and_reference_reordering() -> None:
    adapter = DA3Model.__new__(DA3Model)
    adapter.resolution = 32

    class API:
        def inference(self, images: Sequence[Array], **kwargs: object) -> SimpleNamespace:
            assert kwargs["ref_view_strategy"] == "first"
            x = np.array([image[0, 0, 0] for image in images], float)
            extrinsics = np.repeat(np.eye(4)[None, :3], len(images), 0)
            extrinsics[:, 0, 3] = -x
            return SimpleNamespace(
                extrinsics=extrinsics,
                depth=np.ones((len(images), 4, 4)),
                conf=np.ones((len(images), 4, 4)),
                intrinsics=np.repeat(np.eye(3)[None], len(images), 0),
            )

    frames = [Frame(i, float(i), np.full((4, 4, 3), i, np.uint8)) for i in range(3)]
    with patch.object(adapter, "model", API(), create=True):
        result = adapter.reconstruct(frames, reference=1)
    np.testing.assert_allclose(result.poses[:, 0, 3], [-1, 0, 1])
