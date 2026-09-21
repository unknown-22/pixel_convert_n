import asyncio
import unittest
from dataclasses import replace

import numpy as np
from PIL import Image

from pixel_art_logic import (
    DitheringType,
    FilterType,
    PixelArtConfig,
    ResizeMethod,
    process_image,
)


class PixelArtTests(unittest.TestCase):
    def convert(self, image, **options):
        return asyncio.run(process_image(image, PixelArtConfig(**options)))

    def test_nearest_keeps_source_colors_and_output_matches_small(self):
        image = np.random.default_rng(4).integers(0, 256, (12, 16, 3), dtype=np.uint8)
        result, small = self.convert(image, scale_factor=0.25, apply_kmeans=False)
        self.assertEqual(small.shape, (3, 4, 4))
        source_colors = {tuple(pixel) for pixel in image.reshape(-1, 3)}
        self.assertTrue(
            all(
                tuple(pixel) in source_colors for pixel in small[..., :3].reshape(-1, 3)
            )
        )
        np.testing.assert_array_equal(
            result, np.repeat(np.repeat(small, 4, axis=0), 4, axis=1)
        )

    def test_half_transparent_color_is_restored_before_threshold(self):
        image = np.full((8, 8, 4), [200, 100, 50, 192], dtype=np.uint8)
        for method in ResizeMethod:
            for filter_type in FilterType:
                with self.subTest(method=method, filter_type=filter_type):
                    _, small = self.convert(
                        image,
                        resize_method=method,
                        filter_type=filter_type,
                        scale_factor=0.5,
                    )
                    np.testing.assert_array_equal(
                        small, np.full_like(small, [200, 100, 50, 255])
                    )

    def test_hidden_rgb_does_not_change_visible_result(self):
        image = np.zeros((12, 12, 4), dtype=np.uint8)
        image[3:9, 3:9] = [200, 100, 50, 255]
        alternate = image.copy()
        alternate[image[..., 3] == 0, :3] = [255, 0, 255]
        for method in ResizeMethod:
            for filter_type in FilterType:
                with self.subTest(method=method, filter_type=filter_type):
                    options = {
                        "resize_method": method,
                        "filter_type": filter_type,
                        "apply_erosion": True,
                        "erosion_size": 3,
                        "colors": 1,
                        "scale_factor": 0.5,
                    }
                    _, small = self.convert(image, **options)
                    _, other = self.convert(alternate, **options)
                    np.testing.assert_array_equal(small, other)
                    np.testing.assert_array_equal(small[small[..., 3] == 0], 0)
                    np.testing.assert_allclose(
                        small[small[..., 3] > 0, :3],
                        np.full_like(
                            small[small[..., 3] > 0, :3], [200, 100, 50]
                        ),
                        atol=1,
                    )

    def test_empty_palette_and_excess_colors(self):
        for image in [
            np.zeros((1, 1, 4), dtype=np.uint8),
            np.full((1, 1, 4), 255, dtype=np.uint8),
        ]:
            result, small = self.convert(image, colors=32)
            np.testing.assert_array_equal(result, image)
            np.testing.assert_array_equal(small, image)

    def test_float32_input_has_same_brightness_as_uint8(self):
        image = np.full((8, 8, 3), [100, 160, 220], dtype=np.uint8)
        _, integer = self.convert(image)
        _, floating = self.convert(image.astype(np.float32) / 255)
        np.testing.assert_array_equal(integer, floating)

    def test_quantization_color_limit_and_non_divisible_output(self):
        image = np.random.default_rng(2).integers(0, 256, (13, 17, 3), dtype=np.uint8)
        result, small = self.convert(image, colors=3, scale_factor=0.25)
        self.assertLessEqual(len(np.unique(small.reshape(-1, 4), axis=0)), 3)
        expected = np.asarray(
            Image.fromarray(small).resize((17, 13), Image.Resampling.NEAREST)
        )
        np.testing.assert_array_equal(result, expected)

    def test_ordered_dithering_uses_only_palette_colors_and_is_deterministic(self):
        ramp = np.linspace(0, 255, 16, dtype=np.uint8)
        image = np.repeat(ramp[None, :, None], 16, axis=0)
        image = np.repeat(image, 3, axis=2)
        options = {
            "colors": 3,
            "scale_factor": 1,
            "dithering_type": DitheringType.ORDERED,
            "dithering_strength": 0.2,
        }
        _, dithered = self.convert(image, **options)
        _, repeated = self.convert(image, **options)
        _, plain = self.convert(image, colors=3, scale_factor=1)

        np.testing.assert_array_equal(dithered, repeated)
        self.assertFalse(np.array_equal(dithered, plain))
        self.assertLessEqual(
            len(np.unique(dithered[..., :3].reshape(-1, 3), axis=0)), 3
        )

    def test_zero_strength_dithering_matches_normal_quantization(self):
        image = np.random.default_rng(7).integers(0, 256, (12, 12, 3), dtype=np.uint8)
        _, plain = self.convert(image, colors=4)
        _, dithered = self.convert(
            image,
            colors=4,
            dithering_type=DitheringType.ORDERED,
            dithering_strength=0,
        )
        np.testing.assert_array_equal(dithered, plain)

    def test_bilateral_preserves_edge_while_smoothing_noise(self):
        rng = np.random.default_rng(0)
        image = np.zeros((20, 20, 3), dtype=np.float32)
        image[:, :10] = 0.2
        image[:, 10:] = 0.8
        image += rng.normal(0, 0.02, image.shape).astype(np.float32)
        _, result = self.convert(
            image, filter_type=FilterType.BILATERAL, scale_factor=1, apply_kmeans=False
        )
        rgb = result[..., :3] / 255
        self.assertLess(rgb[3:-3, :8].std(), image[3:-3, :8].std())
        self.assertGreater(rgb[:, 10].mean() - rgb[:, 9].mean(), 0.5)

    def test_erosion_can_be_combined_with_smoothing(self):
        image = np.full((9, 9, 3), 220, dtype=np.uint8)
        image[4, 4] = 30
        for filter_type in [FilterType.GAUSSIAN, FilterType.BILATERAL]:
            config = PixelArtConfig(
                scale_factor=1,
                apply_kmeans=False,
                filter_type=filter_type,
                erosion_size=3,
            )
            _, plain = asyncio.run(process_image(image, config))
            _, eroded = asyncio.run(
                process_image(image, replace(config, apply_erosion=True))
            )
            self.assertLess(eroded[..., :3].sum(), plain[..., :3].sum())


if __name__ == "__main__":
    unittest.main()
