from __future__ import annotations

import argparse
import asyncio
from pathlib import Path

import numpy as np
from PIL import Image

from pixel_art_logic import (
    FilterType,
    PixelArtConfig,
    SaturationLevel,
    pixel_art_converter,
)


def _filter_enum_to_cli(value: FilterType) -> str:
    match value:
        case FilterType.GAUSSIAN:
            return "gaussian"
        case FilterType.EROSION:
            return "erosion"
        case _:
            return "none"


def _saturation_enum_to_cli(value: SaturationLevel) -> str:
    match value:
        case SaturationLevel.WEAK:
            return "weak"
        case SaturationLevel.STRONG:
            return "strong"
        case _:
            return "none"


def _filter_cli_to_label(value: str) -> str:
    match value:
        case "gaussian":
            return "ガウシアンフィルタ"
        case "erosion":
            return "エロージョン"
        case _:
            return ""


def _saturation_cli_to_label(value: str) -> str:
    match value:
        case "weak":
            return "弱"
        case "strong":
            return "強"
        case _:
            return ""


def build_parser() -> argparse.ArgumentParser:
    defaults = PixelArtConfig()
    parser = argparse.ArgumentParser(
        description="指定ディレクトリ内のPNGをドット絵化して *_converted.png として保存します。",
    )
    parser.add_argument(
        "directory",
        type=Path,
        help="PNGを再帰的に探索するディレクトリ",
    )
    parser.add_argument(
        "--scale-factor",
        type=float,
        default=defaults.scale_factor,
        help="縮小率 (例: 0.15)",
    )
    parser.add_argument(
        "--colors",
        type=int,
        default=defaults.colors,
        help="減色後の色数",
    )
    parser.add_argument(
        "--filter-type",
        choices=("none", "gaussian", "erosion"),
        default=_filter_enum_to_cli(defaults.filter_type),
        help="フィルター種別 (none/gaussian/erosion)",
    )
    parser.add_argument(
        "--gaussian-sigma",
        type=float,
        default=defaults.gaussian_sigma,
        help="ガウシアンフィルタのシグマ値",
    )
    parser.add_argument(
        "--erosion-size",
        type=int,
        default=defaults.erosion_size,
        help="エロージョンのカーネルサイズ",
    )
    parser.add_argument(
        "--apply-kmeans",
        action=argparse.BooleanOptionalAction,
        default=defaults.apply_kmeans,
        help="k-meansによる減色を適用するか",
    )
    parser.add_argument(
        "--saturation-level",
        choices=("none", "weak", "strong"),
        default=_saturation_enum_to_cli(defaults.saturation_level),
        help="彩度調整の強さ (none/weak/strong)",
    )
    parser.add_argument(
        "--apply-color-temperature",
        action=argparse.BooleanOptionalAction,
        default=defaults.apply_color_temperature,
        help="色温度調整を適用するか",
    )
    parser.add_argument(
        "--color-temperature-offset",
        type=int,
        default=defaults.color_temperature_offset,
        help="色温度オフセット (-35〜35 推奨)",
    )
    return parser


def list_png_files(root: Path) -> list[Path]:
    return [path for path in root.rglob("*.png") if path.is_file()]


async def convert_one(path: Path, args: argparse.Namespace) -> Path:
    with Image.open(path) as img:
        input_array = np.array(img.convert("RGBA"))

    _, small_array = await pixel_art_converter(
        input_array,
        scale_factor=args.scale_factor,
        colors=args.colors,
        filter_type=_filter_cli_to_label(args.filter_type),
        gaussian_sigma=args.gaussian_sigma,
        erosion_size=args.erosion_size,
        apply_kmeans=args.apply_kmeans,
        saturation_level=_saturation_cli_to_label(args.saturation_level),
        apply_color_temperature=args.apply_color_temperature,
        color_temperature_offset=args.color_temperature_offset,
    )

    output_path = path.with_name(f"{path.stem}_converted{path.suffix}")
    Image.fromarray(small_array).save(output_path)
    return output_path


async def run(args: argparse.Namespace) -> None:
    target_dir = args.directory
    if not target_dir.is_dir():
        msg = f"指定パスがディレクトリではありません: {target_dir}"
        raise NotADirectoryError(msg)

    png_files = list_png_files(target_dir)
    if not png_files:
        print("PNGファイルが見つかりませんでした。")
        return

    for path in png_files:
        try:
            output_path = await convert_one(path, args)
            print(f"converted: {path} -> {output_path}")
        except Exception as exc:
            print(f"failed: {path} ({exc})")


def main() -> None:
    parser = build_parser()
    args = parser.parse_args()
    asyncio.run(run(args))


if __name__ == "__main__":
    main()
