from __future__ import annotations

import argparse
import asyncio
from pathlib import Path

import numpy as np
from PIL import Image

from pixel_art_logic import (
    MAX_COLORS,
    DitheringType,
    FilterType,
    PixelArtConfig,
    ResizeMethod,
    SaturationLevel,
    process_image,
)


def _positive_int(value: str) -> int:
    parsed = int(value)
    if parsed < 1:
        raise argparse.ArgumentTypeError("1以上を指定してください")
    return parsed


def _color_count(value: str) -> int:
    parsed = int(value)
    if not 1 <= parsed <= MAX_COLORS:
        raise argparse.ArgumentTypeError(f"1以上{MAX_COLORS}以下を指定してください")
    return parsed


def _temperature_offset(value: str) -> int:
    parsed = int(value)
    if not -35 <= parsed <= 35:
        raise argparse.ArgumentTypeError("-35以上35以下を指定してください")
    return parsed


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
        type=_color_count,
        default=defaults.colors,
        help="減色後の色数",
    )
    parser.add_argument(
        "--filter-type",
        choices=("none", "gaussian", "bilateral", "erosion"),
        default=defaults.filter_type.value,
        help="フィルター種別 (none/gaussian/bilateral/erosion)",
    )
    parser.add_argument(
        "--gaussian-sigma",
        type=float,
        default=defaults.gaussian_sigma,
        help="ガウシアンフィルタのシグマ値",
    )
    parser.add_argument(
        "--erosion-size",
        type=_positive_int,
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
        default=defaults.saturation_level.value,
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
        type=_temperature_offset,
        default=defaults.color_temperature_offset,
        help="色温度オフセット (-35〜35)",
    )
    parser.add_argument(
        "--resize-method",
        choices=("nearest", "lanczos"),
        default=defaults.resize_method.value,
        help="縮小方法（既定: nearest）",
    )
    parser.add_argument(
        "--apply-erosion",
        action=argparse.BooleanOptionalAction,
        default=defaults.apply_erosion,
        help="平滑化の前にエロージョンを適用",
    )
    parser.add_argument(
        "--bilateral-sigma-color",
        type=float,
        default=defaults.bilateral_sigma_color,
        help="バイラテラルの色差範囲（RGB 0〜1単位）",
    )
    parser.add_argument(
        "--bilateral-sigma-spatial",
        type=float,
        default=defaults.bilateral_sigma_spatial,
        help="バイラテラルの距離範囲（入力画像のピクセル単位）",
    )
    parser.add_argument(
        "--dithering",
        choices=("none", "ordered"),
        default=defaults.dithering_type.value,
        help="ディザリング方式 (none/ordered)",
    )
    parser.add_argument(
        "--dithering-strength",
        type=float,
        default=defaults.dithering_strength,
        help="orderedディザリングの強度 (0〜1、推奨0.08〜0.15)",
    )
    parser.add_argument(
        "--overwrite",
        action="store_true",
        help="既存の *_converted.png を上書きする",
    )
    return parser


def list_png_files(root: Path) -> list[Path]:
    return sorted(
        path
        for path in root.rglob("*.png")
        if path.is_file() and not path.stem.endswith("_converted")
    )


async def convert_one(path: Path, args: argparse.Namespace) -> Path | None:
    output_path = path.with_name(f"{path.stem}_converted{path.suffix}")
    if output_path.exists() and not args.overwrite:
        return None

    with Image.open(path) as img:
        input_array = np.array(img.convert("RGBA"))

    config = PixelArtConfig(
        resize_method=ResizeMethod(args.resize_method),
        apply_erosion=args.apply_erosion,
        bilateral_sigma_color=args.bilateral_sigma_color,
        bilateral_sigma_spatial=args.bilateral_sigma_spatial,
        scale_factor=args.scale_factor,
        colors=args.colors,
        filter_type=FilterType(args.filter_type),
        gaussian_sigma=args.gaussian_sigma,
        erosion_size=args.erosion_size,
        apply_kmeans=args.apply_kmeans,
        saturation_level=SaturationLevel(args.saturation_level),
        apply_color_temperature=args.apply_color_temperature,
        color_temperature_offset=args.color_temperature_offset,
        dithering_type=DitheringType(args.dithering),
        dithering_strength=args.dithering_strength,
    )
    _, small_array = await process_image(input_array, config)

    Image.fromarray(small_array).save(output_path)
    return output_path


async def run(args: argparse.Namespace) -> int:
    target_dir = args.directory
    if not target_dir.is_dir():
        msg = f"指定パスがディレクトリではありません: {target_dir}"
        raise NotADirectoryError(msg)

    png_files = list_png_files(target_dir)
    if not png_files:
        print("PNGファイルが見つかりませんでした。")
        return 0

    converted = 0
    skipped = 0
    failed = 0
    for path in png_files:
        try:
            output_path = await convert_one(path, args)
            if output_path is None:
                skipped += 1
                print(f"skipped: {path} (出力ファイルが存在します)")
            else:
                converted += 1
                print(f"converted: {path} -> {output_path}")
        except (OSError, RuntimeError, ValueError) as exc:
            failed += 1
            print(f"failed: {path} ({exc})")
    print(f"summary: converted={converted}, skipped={skipped}, failed={failed}")
    return 1 if failed else 0


def main() -> None:
    parser = build_parser()
    args = parser.parse_args()
    raise SystemExit(asyncio.run(run(args)))


if __name__ == "__main__":
    main()
