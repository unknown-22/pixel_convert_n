import argparse
import asyncio
import tempfile
import unittest
from pathlib import Path

import numpy as np
from PIL import Image

from batch_pixel_art_converter import build_parser, list_png_files, run


class BatchPixelArtConverterTests(unittest.TestCase):
    def parse_args(self, directory: Path, *options: str) -> argparse.Namespace:
        return build_parser().parse_args([str(directory), *options])

    def test_list_png_files_excludes_converted_outputs(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            (root / "source.png").touch()
            (root / "source_converted.png").touch()
            (root / "nested").mkdir()
            (root / "nested" / "other.png").touch()

            self.assertEqual(
                list_png_files(root),
                [root / "nested" / "other.png", root / "source.png"],
            )

    def test_existing_output_is_skipped_unless_overwrite_is_enabled(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            source = root / "source.png"
            output = root / "source_converted.png"
            Image.fromarray(np.full((4, 4, 3), 100, dtype=np.uint8)).save(source)
            Image.fromarray(np.full((1, 1, 3), 25, dtype=np.uint8)).save(output)

            exit_code = asyncio.run(run(self.parse_args(root)))
            self.assertEqual(exit_code, 0)
            self.assertEqual(Image.open(output).size, (1, 1))

            exit_code = asyncio.run(run(self.parse_args(root, "--overwrite")))
            self.assertEqual(exit_code, 0)
            self.assertEqual(Image.open(output).size, (1, 1))
            self.assertEqual(np.asarray(Image.open(output))[0, 0, 0], 100)
            self.assertFalse((root / "source_converted_converted.png").exists())

    def test_failed_image_returns_nonzero_exit_code(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            (root / "broken.png").write_text("not a png", encoding="utf-8")

            self.assertEqual(asyncio.run(run(self.parse_args(root))), 1)

    def test_temperature_argument_is_bounded(self):
        with (
            tempfile.TemporaryDirectory() as directory,
            self.assertRaises(SystemExit),
        ):
            self.parse_args(Path(directory), "--color-temperature-offset", "36")

    def test_color_count_is_bounded(self):
        with (
            tempfile.TemporaryDirectory() as directory,
            self.assertRaises(SystemExit),
        ):
            self.parse_args(Path(directory), "--colors", "257")


if __name__ == "__main__":
    unittest.main()
