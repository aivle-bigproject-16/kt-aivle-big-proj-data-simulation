from __future__ import annotations

import os
import unittest
from pathlib import Path

import numpy as np
from PIL import Image

from server_sim_dataset.generator import _failure_engine, _mask


ENGINE_ROOT_VALUE = os.environ.get("QUALITY_FAIL_ENGINE_ROOT")
ENGINE_ROOT = Path(ENGINE_ROOT_VALUE) if ENGINE_ROOT_VALUE else None


@unittest.skipUnless(
    ENGINE_ROOT is not None and ENGINE_ROOT.is_dir(),
    "QUALITY_FAIL_ENGINE_ROOT is not configured",
)
class FailureEngineIntegrationTests(unittest.TestCase):
    def test_v2_rgb_failure_adapter(self) -> None:
        assert ENGINE_ROOT is not None
        engine = _failure_engine(ENGINE_ROOT)
        gradient = np.tile(np.arange(256, dtype=np.uint8), (256, 1))
        image = Image.merge("RGB", tuple(Image.fromarray(gradient) for _ in range(3)))
        outline = [(24, 24), (232, 24), (232, 232), (24, 232)]
        result = engine.apply_failure_case(
            image,
            "RGB",
            "rgb_underexposure",
            20260723,
            _mask(image.size, [outline]),
        )
        self.assertEqual(result.image.size, image.size)
        self.assertTrue(result.records)

