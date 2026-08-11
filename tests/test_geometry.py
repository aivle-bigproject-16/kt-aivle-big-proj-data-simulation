"""계획서 v1.5의 CT 3-D 변환 및 정상 증강 회귀 테스트."""

from __future__ import annotations

import inspect
import unittest

from PIL import Image, ImageOps

from server_sim_dataset import generator
from server_sim_dataset.schema import ct_axis_transform, ct_view_to_voxel
from server_sim_dataset.util import stable_seed


def _grayscale(width: int = 64, height: int = 96) -> Image.Image:
    image = Image.new("L", (width, height), 40)
    for x in range(width // 3):
        for y in range(height // 4):
            image.putpixel((x, y), 220)
    return image


PARAMS = {"brightness": 1.0, "contrast": 1.0, "gamma": 1.0, "noise_sigma": 0.003}


class NormalApiTests(unittest.TestCase):
    def test_normal_separates_slice_seed_from_id_seed(self) -> None:
        """F-01, 계획서 6.2 와 13.5 의 6 항.

        슬라이스마다 달라져야 하는 효과와 ID 전체에 동기 적용되어야 하는 효과는 서로
        다른 seed 를 써야 한다. seed 를 하나만 받는 시그니처로는 두 요구를 동시에
        만족시킬 수 없다.
        """
        signature = inspect.signature(generator._normal)
        self.assertIn(
            "id_seed",
            signature.parameters,
            "_normal 이 ID 단위 seed 를 별도로 받아야 한다",
        )
        self.assertIn("slice_seed", signature.parameters)


class SynchronizedFlipTests(unittest.TestCase):
    """F-01. v1.2 에서 CT 4,350 장의 볼륨 방향이 슬라이스마다 뒤바뀐 결함."""

    def _flip_of(self, slice_seed: int, id_seed: int, axis: str = "x") -> tuple[bool, bool]:
        result = generator._normal(
            _grayscale(), ["synchronized_flip"], PARAMS, slice_seed, id_seed, axis=axis
        )
        return result.flip_horizontal, result.flip_vertical

    def test_flip_is_constant_across_slices_of_one_id(self) -> None:
        """계획서 6.2: 반전 여부와 방향은 ID 별로 한 번만 결정한다."""
        id_seed = stable_seed(20260723, "CT", 101, "normal-base")
        for axis in ("x", "y", "z"):
            observed = {
                self._flip_of(
                    stable_seed(id_seed, axis, index, "synchronized_flip"),
                    id_seed,
                    axis,
                )
                for index in range(40)
            }
            self.assertEqual(len(observed), 1, f"{axis} axis changed between slices")

    def test_flip_differs_between_ids(self) -> None:
        """ID 단위로 고정하되 ID 사이에서는 달라져야 결정론적 다양성이 유지된다."""
        observed = {
            self._flip_of(
                stable_seed(stable_seed(20260723, "CT", battery_id, "normal-base"), "x", 0),
                stable_seed(20260723, "CT", battery_id, "normal-base"),
            )
            for battery_id in range(101, 141)
        }
        self.assertGreater(len(observed), 1, "모든 ID 가 같은 방향으로 고정되었다")

    def test_one_3d_flip_is_projected_consistently_to_every_axis(self) -> None:
        expected = {
            "x": (True, True, True),
            "y": (True, True, True),
            "z": (True, True, True),
        }
        for axis, values in expected.items():
            transform = ct_axis_transform(7, axis)
            self.assertEqual(
                (transform.flip_horizontal, transform.flip_vertical, transform.reverse_slices),
                values,
            )

    def test_global_x_reflection_changes_only_matching_coordinates(self) -> None:
        expected = {
            "x": (False, False, True),
            "y": (True, False, False),
            "z": (True, False, False),
        }
        for axis, values in expected.items():
            transform = ct_axis_transform(1, axis)
            self.assertEqual(
                (transform.flip_horizontal, transform.flip_vertical, transform.reverse_slices),
                values,
            )

    def test_each_view_maps_back_to_the_same_3d_point(self) -> None:
        self.assertEqual(ct_view_to_voxel("x", 10, 20, 30), (10, 20, 30))
        self.assertEqual(ct_view_to_voxel("y", 20, 10, 30), (10, 20, 30))
        self.assertEqual(ct_view_to_voxel("z", 30, 10, 20), (10, 20, 30))

    def test_synchronized_flip_never_resolves_to_the_identity(self) -> None:
        transforms = [ct_axis_transform(0, axis) for axis in ("x", "y", "z")]
        self.assertTrue(
            any(
                transform.flip_horizontal
                or transform.flip_vertical
                or transform.reverse_slices
                for transform in transforms
            )
        )

    def test_all_global_reflections_project_to_one_shared_voxel(self) -> None:
        point = (2.0, 3.0, 4.0)
        limits = (10.0, 20.0, 30.0)
        views = {
            "x": (point[0], point[1], point[2], limits[0], limits[1], limits[2]),
            "y": (point[1], point[0], point[2], limits[1], limits[0], limits[2]),
            "z": (point[2], point[0], point[1], limits[2], limits[0], limits[1]),
        }
        for mask in range(1, 8):
            expected = tuple(
                limit - value if mask & (1 << coordinate) else value
                for coordinate, (value, limit) in enumerate(zip(point, limits))
            )
            for axis, (slice_pos, horizontal, vertical, slice_max, width, height) in views.items():
                transform = ct_axis_transform(mask, axis)
                if transform.reverse_slices:
                    slice_pos = slice_max - slice_pos
                if transform.flip_horizontal:
                    horizontal = width - horizontal
                if transform.flip_vertical:
                    vertical = height - vertical
                self.assertEqual(
                    ct_view_to_voxel(axis, slice_pos, horizontal, vertical),
                    expected,
                    (mask, axis),
                )


class PolygonTransformTests(unittest.TestCase):
    def test_flip_matches_pillow_pixel_mapping(self) -> None:
        """F-11, 계획서 13.3.

        PIL 의 mirror 는 x 를 W-1-x 로 옮긴다. 좌표를 W-x 로 옮기면 반전된 모든
        슬라이스에 1 픽셀 계통 오차가 남는다.
        """
        width, height = 64, 96
        polygon = [(0.0, 0.0), (10.0, 0.0), (10.0, 10.0)]
        flipped = generator._transform_polygon(polygon, width, height, flip_x=True)
        self.assertEqual(flipped[0][0], float(width - 1))
        self.assertEqual(flipped[1][0], float(width - 1 - 10))

        flipped_y = generator._transform_polygon(polygon, width, height, flip_y=True)
        self.assertEqual(flipped_y[0][1], float(height - 1))
        self.assertEqual(flipped_y[1][1], float(height - 1))

    def test_flipped_polygon_tracks_the_flipped_image(self) -> None:
        """이미지와 폴리곤이 같은 변환을 받았는지 실제 픽셀로 확인한다."""
        image = _grayscale()
        width, height = image.size
        mirrored = ImageOps.mirror(image)
        bright_original = [x for x in range(width) if image.getpixel((x, 0)) > 128]
        bright_mirrored = [x for x in range(width) if mirrored.getpixel((x, 0)) > 128]
        mapped = [
            generator._transform_polygon(
                [(float(x), 1.0), (float(x), 6.0), (float(x) + 0.5, 3.0)],
                width,
                height,
                flip_x=True,
            )[0][0]
            for x in bright_original
        ]
        self.assertEqual(sorted(mapped), sorted(float(x) for x in bright_mirrored))


class EnvelopeTests(unittest.TestCase):
    """F-12, 계획서 6.2 와 13.5 의 6 항."""

    def test_envelope_exists(self) -> None:
        self.assertTrue(
            hasattr(generator, "_envelope"),
            "슬라이스 강도의 연속 envelope 이 구현되어야 한다",
        )

    def test_envelope_is_continuous_between_adjacent_slices(self) -> None:
        low, high, length = 0.002, 0.008, 650
        values = generator._envelope(stable_seed("id"), "z", length, low, high)
        self.assertEqual(len(values), length)
        self.assertTrue(all(low <= value <= high for value in values))
        largest_step = max(abs(b - a) for a, b in zip(values, values[1:]))
        self.assertLess(
            largest_step,
            (high - low) / 20,
            "인접 슬라이스 사이 강도 변화가 연속적이지 않다",
        )


class RgbGeometryTests(unittest.TestCase):
    """F-03, 계획서 6.3."""

    def test_safe_translate_rotate_applies_a_real_affine(self) -> None:
        image = Image.new("RGB", (192, 108), (30, 30, 30))
        result = generator._normal(
            image, ["safe_translate_rotate"], PARAMS, stable_seed("slice"), stable_seed("id")
        )
        self.assertIsNotNone(
            getattr(result, "affine", None),
            "이동·회전이 실제 affine 으로 적용되고 그 값이 반환되어야 한다",
        )

    def test_padding_gate_records_its_reason_when_it_falls_back(self) -> None:
        """계획서 6.3 의 5 항: 재시도 및 대체 이유를 manifest 에 기록한다."""
        image = Image.new("RGB", (192, 108), (30, 30, 30))
        result = generator._normal(
            image, ["safe_translate_rotate"], PARAMS, stable_seed("slice"), stable_seed("id")
        )
        self.assertTrue(
            hasattr(result, "retry_reason"),
            "대체 사유를 기록할 필드가 없다",
        )


if __name__ == "__main__":
    unittest.main()
