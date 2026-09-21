import asyncio
from dataclasses import dataclass
from enum import StrEnum

import numpy as np
from PIL import Image, ImageEnhance
from skimage import morphology
from skimage.filters import gaussian
from skimage.restoration import denoise_bilateral
from sklearn.cluster import KMeans


class SaturationLevel(StrEnum):
    """彩度調整レベルの列挙型"""

    NONE = "none"
    WEAK = "weak"
    STRONG = "strong"


class FilterType(StrEnum):
    """フィルタータイプの列挙型"""

    NONE = "none"
    GAUSSIAN = "gaussian"
    EROSION = "erosion"  # 既存の設定との互換性を維持
    BILATERAL = "bilateral"


class ResizeMethod(StrEnum):
    """縮小時の補間方式。拡大には常に最近傍を使う。"""

    NEAREST = "nearest"
    LANCZOS = "lanczos"


class DitheringType(StrEnum):
    """減色時のディザリング方式。"""

    NONE = "none"
    ORDERED = "ordered"


@dataclass
class PixelArtConfig:
    """ドット絵変換の設定パラメータ"""

    scale_factor: float = 0.2  # 縮小率
    colors: int = 8  # 色数
    filter_type: FilterType = FilterType.NONE
    gaussian_sigma: float = 1.0  # ガウシアンフィルタのシグマ値
    erosion_size: int = 1  # エロージョンのカーネルサイズ
    apply_kmeans: bool = True  # k-meansによる減色を適用するかどうか
    saturation_level: SaturationLevel = SaturationLevel.NONE  # 彩度調整レベル
    apply_color_temperature: bool = False  # 色温度調整を適用するかどうか
    color_temperature_offset: int = 0  # 色温度オフセット（0を基準として±100K単位）

    resize_method: ResizeMethod = ResizeMethod.NEAREST
    apply_erosion: bool = False
    bilateral_sigma_color: float = 0.1  # RGB [0, 1] に対する色差
    bilateral_sigma_spatial: float = 3.0  # 入力画像上のピクセル単位
    dithering_type: DitheringType = DitheringType.NONE
    dithering_strength: float = 0.1  # RGB [0, 1] に加える明度差の最大幅


MAX_KMEANS_SAMPLES = 100_000
MAX_COLORS = 256
MAX_DISTANCE_ELEMENTS = 1_000_000


def _temperature_rgb(temperature: int) -> np.ndarray:
    """色温度をRGB係数に変換する。"""
    temp = max(3000, min(10000, temperature)) / 100

    if temp <= 66:
        red = 255
    else:
        red = 329.698727446 * ((temp - 60) ** -0.1332047592)

    if temp <= 66:
        green = 99.4708025861 * np.log(temp) - 161.1195681661
    else:
        green = 288.1221695283 * ((temp - 60) ** -0.0755148492)

    if temp >= 66:
        blue = 255
    elif temp <= 19:
        blue = 0
    else:
        blue = 138.5177312231 * np.log(temp - 10) - 305.0447927307

    return np.clip(np.array([red, green, blue], dtype=np.float32), 0, 255) / 255


def adjust_color_temperature(image: np.ndarray, temperature_offset: int) -> np.ndarray:
    """
    画像の色温度を調整する

    Parameters
    ----------
    image : np.ndarray
        調整する画像（RGB形式）
    temperature_offset : int
        色温度オフセット（0を基準として±100K単位で指定）
        プラス値: 暖色系（色温度を下げる）
        マイナス値: 寒色系（色温度を上げる）

    Returns
    -------
    np.ndarray
        色温度調整後の画像
    """
    base_temperature = 6500
    temperature = base_temperature - (temperature_offset * 100)
    # 6500Kの係数を基準にすることで、オフセット0を完全な無変換にする。
    factors = _temperature_rgb(temperature) / _temperature_rgb(base_temperature)
    return np.clip(image * factors, 0, 1)


def _split_alpha(image: np.ndarray) -> tuple[np.ndarray, np.ndarray | None]:
    """
    画像をRGBとアルファに分離する

    Parameters
    ----------
    image : np.ndarray
        入力画像

    Returns
    -------
    tuple[np.ndarray, np.ndarray | None]
        (RGB画像, アルファチャネルまたはNone)
    """
    if image.ndim == 3 and image.shape[2] == 4:
        rgb, alpha = image[:, :, :3], image[:, :, 3]
        return rgb, alpha
    return image, None


def _bilateral_filter(
    rgb: np.ndarray, alpha: np.ndarray, sigma_color: float, sigma_spatial: float
) -> np.ndarray:
    """乗算済みRGBAをまとめて平滑化し、透明画素の隠れたRGBの混入を防ぐ。"""
    premultiplied_rgba = np.dstack((rgb * alpha[..., None], alpha))
    filtered = denoise_bilateral(
        premultiplied_rgba,
        win_size=15,
        sigma_color=sigma_color,
        sigma_spatial=sigma_spatial,
        bins=1_000,
        mode="edge",
        channel_axis=-1,
    ).astype(np.float32, copy=False)
    filtered_alpha = filtered[..., 3]
    return np.divide(
        filtered[..., :3],
        filtered_alpha[..., None],
        out=rgb.copy(),
        where=filtered_alpha[..., None] > 1e-8,
    )


def _apply_filters(
    rgb: np.ndarray, alpha: np.ndarray, config: PixelArtConfig
) -> np.ndarray:
    """エロージョン→平滑化。透明画素の隠れたRGBは参照しない。"""
    if config.apply_erosion or config.filter_type == FilterType.EROSION:
        footprint = morphology.footprint_rectangle(
            (config.erosion_size, config.erosion_size)
        )
        visible_rgb = np.where(alpha[..., None] > 0, rgb, np.inf)
        rgb = np.stack(
            [
                morphology.erosion(visible_rgb[..., channel], footprint)
                for channel in range(3)
            ],
            axis=-1,
        )
        rgb = np.where(np.isfinite(rgb), rgb, 0)
    if config.filter_type == FilterType.GAUSSIAN:
        numerator = gaussian(
            rgb * alpha[..., None], sigma=config.gaussian_sigma, channel_axis=-1
        )
        denominator = gaussian(alpha, sigma=config.gaussian_sigma)
        rgb = np.divide(
            numerator,
            denominator[..., None],
            out=np.zeros_like(rgb),
            where=denominator[..., None] > 1e-8,
        )
    elif config.filter_type == FilterType.BILATERAL:
        rgb = _bilateral_filter(
            rgb, alpha, config.bilateral_sigma_color, config.bilateral_sigma_spatial
        )
    return np.where(alpha[..., None] > 0, rgb, 0)


def _resize_rgba(
    rgb: np.ndarray, alpha: np.ndarray, size: tuple[int, int], method: ResizeMethod
) -> tuple[np.ndarray, np.ndarray]:
    """乗算済みRGBとアルファを個別にリサイズし、二値化前に色を復元する。"""
    resampling = Image.Resampling[method.name]

    def resize_plane(plane: np.ndarray) -> np.ndarray:
        return np.asarray(
            Image.fromarray(plane.astype(np.float32)).resize(size, resampling)
        )

    # LANCZOSのオーバーシュートもRGBと同じ比率で割り戻す。
    # 先にアルファだけをクランプすると透明境界の色が明るくなる。
    small_alpha = resize_plane(alpha)
    premultiplied = np.stack(
        [resize_plane(rgb[..., channel] * alpha) for channel in range(3)], axis=-1
    )
    small_rgb = np.divide(
        premultiplied,
        small_alpha[..., None],
        out=np.zeros_like(premultiplied),
        where=small_alpha[..., None] > 1e-8,
    )
    return np.clip(small_rgb, 0, 1), small_alpha > 0.5


BAYER_4X4 = (
    np.array(
        [
            [0, 8, 2, 10],
            [12, 4, 14, 6],
            [3, 11, 1, 9],
            [15, 7, 13, 5],
        ],
        dtype=np.float32,
    )
    + 0.5
) / 16 - 0.5


def _create_palette(rgb: np.ndarray, visible: np.ndarray, colors: int) -> np.ndarray:
    """縮小画像の可視画素のみでK-meansパレットを学習する。"""
    pixels = rgb[visible]
    if not len(pixels):
        return np.empty((0, 3), dtype=np.float32)
    if len(pixels) > MAX_KMEANS_SAMPLES:
        indices = (
            np.arange(MAX_KMEANS_SAMPLES, dtype=np.int64)
            * len(pixels)
            // MAX_KMEANS_SAMPLES
        )
        training_pixels = pixels[indices]
    else:
        training_pixels = pixels
    count = min(colors, len(np.unique(training_pixels, axis=0)))
    kmeans = KMeans(n_clusters=count, random_state=42, n_init=10)
    kmeans.fit(training_pixels)
    return kmeans.cluster_centers_.astype(np.float32)


def _nearest_palette_colors(pixels: np.ndarray, palette: np.ndarray) -> np.ndarray:
    """メモリ使用量を抑えながら各画素を最も近いパレット色へ変換する。"""
    mapped = np.empty_like(pixels)
    chunk_size = max(1, MAX_DISTANCE_ELEMENTS // len(palette))
    for start in range(0, len(pixels), chunk_size):
        chunk = pixels[start : start + chunk_size]
        distances = np.sum((chunk[:, None] - palette[None, :]) ** 2, axis=-1)
        mapped[start : start + chunk_size] = palette[np.argmin(distances, axis=1)]
    return mapped


def _map_to_palette(
    rgb: np.ndarray, visible: np.ndarray, palette: np.ndarray
) -> np.ndarray:
    """可視画素を最も近いパレット色へ置き換える。"""
    result = np.zeros_like(rgb)
    if len(palette):
        result[visible] = _nearest_palette_colors(rgb[visible], palette)
    return result


def _ordered_dither(
    rgb: np.ndarray,
    visible: np.ndarray,
    palette: np.ndarray,
    strength: float,
) -> np.ndarray:
    """4×4 Bayer行列で明度を揺らし、K-meansパレットへ割り当てる。"""
    if len(palette) < 2 or strength == 0:
        return _map_to_palette(rgb, visible, palette)

    height, width = visible.shape
    threshold_map = np.tile(
        BAYER_4X4,
        ((height + 3) // 4, (width + 3) // 4),
    )[:height, :width]
    adjusted = np.clip(rgb + threshold_map[..., None] * strength, 0, 1)
    return _map_to_palette(adjusted, visible, palette)


def _quantize(
    rgb: np.ndarray, visible: np.ndarray, colors: int, config: PixelArtConfig
) -> np.ndarray:
    """K-meansパレットを学習し、指定方式で縮小画像を減色する。"""
    palette = _create_palette(rgb, visible, colors)
    if config.dithering_type == DitheringType.ORDERED:
        return _ordered_dither(rgb, visible, palette, config.dithering_strength)
    return _map_to_palette(rgb, visible, palette)


def _process_image(
    image: np.ndarray, config: PixelArtConfig
) -> tuple[np.ndarray, np.ndarray]:
    """前処理→縮小→減色→最近傍拡大。出力は従来どおりRGBA。"""
    if (
        image is None
        or image.ndim != 3
        or image.shape[2] not in (3, 4)
        or not image.size
    ):
        raise ValueError("RGBまたはRGBA画像を指定してください。")
    if not 0 < config.scale_factor <= 1:
        raise ValueError("縮小率は0より大きく1以下にしてください。")
    if not 1 <= config.colors <= MAX_COLORS:
        raise ValueError(f"色数は1以上{MAX_COLORS}以下にしてください。")
    if config.erosion_size < 1:
        raise ValueError("エロージョンのサイズは1以上にしてください。")
    if not -35 <= config.color_temperature_offset <= 35:
        raise ValueError("色温度オフセットは-35以上35以下にしてください。")
    if any(
        not np.isfinite(value) or value <= 0
        for value in (
            config.gaussian_sigma,
            config.bilateral_sigma_color,
            config.bilateral_sigma_spatial,
        )
    ):
        raise ValueError("フィルターの強さは正の有限値にしてください。")
    if (
        not np.isfinite(config.dithering_strength)
        or not 0 <= config.dithering_strength <= 1
    ):
        raise ValueError("ディザリング強度は0以上1以下にしてください。")
    normalized = image.astype(np.float32)
    if np.issubdtype(image.dtype, np.integer):
        normalized /= np.iinfo(image.dtype).max
    if not np.all(np.isfinite(normalized)):
        raise ValueError("画像に非有限値が含まれています。")
    normalized = np.clip(normalized, 0, 1)
    rgb, alpha = _split_alpha(normalized)
    height, width = rgb.shape[:2]
    if alpha is None:
        alpha = np.ones((height, width), dtype=np.float32)
    if config.apply_color_temperature:
        rgb = adjust_color_temperature(rgb, config.color_temperature_offset)
    if config.saturation_level != SaturationLevel.NONE:
        factor = 1.3 if config.saturation_level == SaturationLevel.WEAK else 1.8
        pil_img = Image.fromarray(np.rint(rgb * 255).astype(np.uint8))
        rgb = (
            np.asarray(ImageEnhance.Color(pil_img).enhance(factor)).astype(np.float32)
            / 255
        )
    rgb = _apply_filters(rgb, alpha, config)
    size = (
        max(1, int(width * config.scale_factor)),
        max(1, int(height * config.scale_factor)),
    )
    rgb, visible = _resize_rgba(rgb, alpha, size, config.resize_method)
    if config.apply_kmeans:
        rgb = _quantize(rgb, visible, config.colors, config)
    rgb = np.where(visible[..., None], rgb, 0)
    small_array = np.dstack(
        (np.rint(rgb * 255).astype(np.uint8), visible.astype(np.uint8) * 255)
    )
    result = np.asarray(
        Image.fromarray(small_array).resize((width, height), Image.Resampling.NEAREST)
    )
    return result, small_array


async def process_image(
    image: np.ndarray, config: PixelArtConfig
) -> tuple[np.ndarray, np.ndarray]:
    """CPU処理を別スレッドで実行し、拡大画像と縮小画像を返す。"""
    return await asyncio.to_thread(_process_image, image, config)


def parse_filter_type(value: FilterType | str) -> FilterType:
    """UIの旧日本語ラベルと安定した機械用値をEnumに変換する。"""
    if isinstance(value, FilterType):
        return value
    legacy_labels = {
        "": FilterType.NONE,
        "なし": FilterType.NONE,
        "ガウシアンフィルタ": FilterType.GAUSSIAN,
        "バイラテラルフィルタ": FilterType.BILATERAL,
        "エロージョン": FilterType.EROSION,
    }
    if value in legacy_labels:
        return legacy_labels[value]
    return FilterType(value)


def parse_saturation_level(value: SaturationLevel | str) -> SaturationLevel:
    """UIの旧日本語ラベルと安定した機械用値をEnumに変換する。"""
    if isinstance(value, SaturationLevel):
        return value
    legacy_labels = {
        "": SaturationLevel.NONE,
        "なし": SaturationLevel.NONE,
        "弱": SaturationLevel.WEAK,
        "強": SaturationLevel.STRONG,
    }
    if value in legacy_labels:
        return legacy_labels[value]
    return SaturationLevel(value)


async def pixel_art_converter(
    input_img: np.ndarray,
    scale_factor: float,
    colors: int,
    filter_type: str,
    gaussian_sigma: float,
    erosion_size: int,
    apply_kmeans: bool,
    saturation_level: str,
    apply_color_temperature: bool,
    color_temperature_offset: int,
    resize_method: str = "nearest",
    apply_erosion: bool = False,
    bilateral_sigma_color: float = 0.1,
    bilateral_sigma_spatial: float = 3.0,
    dithering_type: str = "none",
    dithering_strength: float = 0.1,
) -> tuple[np.ndarray, np.ndarray]:
    """
    UI用のインターフェース関数

    Parameters
    ----------
    input_img : np.ndarray
        入力画像
    scale_factor : float
        縮小率
    colors : int
        色数
    filter_type : str
        フィルタータイプの文字列
    gaussian_sigma : float
        ガウシアンフィルタのシグマ値
    erosion_size : int
        エロージョンのカーネルサイズ
    apply_kmeans : bool
        K-meansによる減色を適用するかどうか
    saturation_level : str
        彩度調整レベルの文字列
    apply_color_temperature : bool
        色温度調整を適用するかどうか
    color_temperature_offset : int
        色温度オフセット（0を基準として±100K単位で指定）

    Returns
    -------
    tuple[np.ndarray, np.ndarray]
        (ドット絵画像, 縮小画像)
    """
    config = PixelArtConfig(
        resize_method=ResizeMethod(resize_method),
        apply_erosion=apply_erosion,
        bilateral_sigma_color=bilateral_sigma_color,
        bilateral_sigma_spatial=bilateral_sigma_spatial,
        dithering_type=DitheringType(dithering_type),
        dithering_strength=dithering_strength,
        scale_factor=scale_factor,
        colors=colors,
        filter_type=parse_filter_type(filter_type),
        gaussian_sigma=gaussian_sigma,
        erosion_size=erosion_size,
        apply_kmeans=apply_kmeans,
        saturation_level=parse_saturation_level(saturation_level),
        apply_color_temperature=apply_color_temperature,
        color_temperature_offset=color_temperature_offset,
    )

    return await process_image(input_img, config)
