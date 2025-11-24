import asyncio
from dataclasses import dataclass
from enum import Enum, auto

import numpy as np
from PIL import Image, ImageEnhance
from skimage import morphology
from skimage.filters import gaussian
from sklearn.cluster import KMeans


class SaturationLevel(Enum):
    """彩度調整レベルの列挙型"""

    NONE = auto()
    WEAK = auto()
    STRONG = auto()


class FilterType(Enum):
    """フィルタータイプの列挙型"""

    NONE = auto()
    GAUSSIAN = auto()
    EROSION = auto()


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
    # ベース色温度は6500Kとし、オフセットを逆向きに適用
    # プラス値で暖色系にするため、色温度を下げる
    base_temperature = 6500
    temperature = base_temperature - (temperature_offset * 100)
    
    # 色温度に基づくRGB係数の計算
    # 参考: http://www.tannerhelland.com/4435/convert-temperature-rgb-algorithm-code/
    
    temp = max(3000, min(10000, temperature))  # 範囲を制限
    temp = temp / 100
    
    # 赤成分の計算
    if temp <= 66:
        red = 255
    else:
        red = temp - 60
        red = 329.698727446 * (red ** -0.1332047592)
        red = max(0, min(255, red))
    
    # 緑成分の計算
    if temp <= 66:
        green = temp
        green = 99.4708025861 * np.log(green) - 161.1195681661
    else:
        green = temp - 60
        green = 288.1221695283 * (green ** -0.0755148492)
    green = max(0, min(255, green))
    
    # 青成分の計算
    if temp >= 66:
        blue = 255
    elif temp <= 19:
        blue = 0
    else:
        blue = temp - 10
        blue = 138.5177312231 * np.log(blue) - 305.0447927307
        blue = max(0, min(255, blue))
    
    # RGB係数を正規化
    red_factor = red / 255.0
    green_factor = green / 255.0
    blue_factor = blue / 255.0
    
    # 画像に色温度補正を適用
    adjusted_image = image.copy()
    if len(adjusted_image.shape) == 3:  # カラー画像の場合
        adjusted_image[:, :, 0] *= red_factor    # R
        adjusted_image[:, :, 1] *= green_factor  # G
        adjusted_image[:, :, 2] *= blue_factor   # B
        
        # 値の範囲をクランプ
        adjusted_image = np.clip(adjusted_image, 0, 1)
    
    return adjusted_image


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


def _merge_alpha(rgb: np.ndarray, alpha: np.ndarray | None) -> np.ndarray:
    """
    RGB画像とアルファチャネルを結合する
    """
    if alpha is None:
        return rgb
    return np.dstack((rgb, alpha))


async def process_image(
    image: np.ndarray, config: PixelArtConfig
) -> tuple[np.ndarray, np.ndarray]:
    """
    画像をドット絵に変換する処理を行う

    Returns
    -------
    tuple[np.ndarray, np.ndarray]
        (ドット絵画像, 縮小画像)
    """
    await asyncio.sleep(0.01)

    # アルファ分離と正規化
    rgb_image, alpha_channel = _split_alpha(image)
    height, width = rgb_image.shape[:2]
    rgb_image = (
        rgb_image.astype(np.float64) / 255.0
        if rgb_image.dtype != np.float64
        else np.clip(rgb_image, 0, 1)
    )
    if alpha_channel is None:
        alpha_channel = np.ones((height, width), dtype=np.float64)
    else:
        alpha_channel = (
            alpha_channel.astype(np.float64) / 255.0
            if alpha_channel.dtype != np.float64
            else np.clip(alpha_channel, 0, 1)
        )

    # 前処理（アルファ非考慮領域はRGB値を持つが後でプリマルチプライする）
    if config.apply_color_temperature:
        rgb_image = adjust_color_temperature(rgb_image, config.color_temperature_offset)

    if config.saturation_level != SaturationLevel.NONE:
        pil_img = Image.fromarray((rgb_image * 255).astype(np.uint8))
        match config.saturation_level:
            case SaturationLevel.WEAK:
                saturation_factor = 1.3
            case SaturationLevel.STRONG:
                saturation_factor = 1.8
            case _:
                saturation_factor = 1.0
        enhancer = ImageEnhance.Color(pil_img)
        pil_img = enhancer.enhance(saturation_factor)
        rgb_image = np.array(pil_img).astype(np.float64) / 255.0

    # プリマルチプライしてからスケーリング・フィルタを適用
    rgb_premult = rgb_image * alpha_channel[..., None]

    match config.filter_type:
        case FilterType.GAUSSIAN:
            rgb_premult = gaussian(
                rgb_premult, sigma=config.gaussian_sigma, channel_axis=-1
            )
        case FilterType.EROSION:
            size = int(config.erosion_size)
            footprint = morphology.footprint_rectangle((size, size))
            for i in range(rgb_premult.shape[2]):
                rgb_premult[:, :, i] = morphology.erosion(
                    rgb_premult[:, :, i], footprint
                )
        case _:
            pass

    # RGBA（プリマルチプライ済みRGB+アルファ）でリサイズ
    pil_img = Image.fromarray(
        np.dstack(
            (
                (np.clip(rgb_premult, 0, 1) * 255).astype(np.uint8),
                (np.clip(alpha_channel, 0, 1) * 255).astype(np.uint8),
            )
        ),
        mode="RGBA",
    )

    new_width = max(1, int(width * config.scale_factor))
    new_height = max(1, int(height * config.scale_factor))
    small_img = pil_img.resize((new_width, new_height), Image.Resampling.LANCZOS)

    small_rgba = np.array(small_img).astype(np.float64) / 255.0
    small_alpha_array = small_rgba[:, :, 3]
    small_array_premult = small_rgba[:, :, :3]

    pixelated_img = small_img.resize((width, height), Image.Resampling.NEAREST)
    pixelated_rgba = np.array(pixelated_img).astype(np.float64) / 255.0
    alpha_result = pixelated_rgba[:, :, 3]
    result_premult = pixelated_rgba[:, :, :3]

    # 透過エッジの白化防止: アルファを二値化し、透明部のRGBを強制ゼロ
    alpha_threshold = 0.5
    alpha_result = (alpha_result > alpha_threshold).astype(np.float64)
    small_alpha_array = (small_alpha_array > alpha_threshold).astype(np.float64)
    result_premult *= alpha_result[..., None]
    small_array_premult *= small_alpha_array[..., None]

    # K-meansはプリマルチプライRGBで実施（アルファは除外）
    if config.apply_kmeans and config.colors > 0:
        original_shape = result_premult.shape
        pixels = result_premult.reshape(-1, 3)

        kmeans = KMeans(n_clusters=config.colors, random_state=42, n_init=10)
        labels = kmeans.fit_predict(pixels)
        centers = kmeans.cluster_centers_

        result_premult = centers[labels].reshape(original_shape)

        small_pixels = small_array_premult.reshape(-1, 3)
        small_labels = kmeans.predict(small_pixels)
        small_array_premult = centers[small_labels].reshape(small_array_premult.shape)

    # アンプリマルチで元の色に戻す（透明周辺の白浮きを防ぐ）
    eps = 1e-8
    alpha_safe = np.maximum(alpha_result, eps)
    result_rgb = np.where(
        alpha_result[..., None] > 0, result_premult / alpha_safe[..., None], 0
    )

    small_alpha_safe = np.maximum(small_alpha_array, eps)
    small_rgb = np.where(
        small_alpha_array[..., None] > 0,
        small_array_premult / small_alpha_safe[..., None],
        0,
    )

    result = _merge_alpha(
        (np.clip(result_rgb, 0, 1) * 255).astype(np.uint8),
        (np.clip(alpha_result, 0, 1) * 255).astype(np.uint8),
    )
    small_array = _merge_alpha(
        (np.clip(small_rgb, 0, 1) * 255).astype(np.uint8),
        (np.clip(small_alpha_array, 0, 1) * 255).astype(np.uint8),
    )

    return result, small_array


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
    # フィルタータイプの文字列をEnum型に変換
    filter_enum = FilterType.NONE
    match filter_type:
        case "ガウシアンフィルタ":
            filter_enum = FilterType.GAUSSIAN
        case "エロージョン":
            filter_enum = FilterType.EROSION
        case _:
            filter_enum = FilterType.NONE

    # 彩度調整レベルの文字列をEnum型に変換
    saturation_enum = SaturationLevel.NONE
    match saturation_level:
        case "弱":
            saturation_enum = SaturationLevel.WEAK
        case "強":
            saturation_enum = SaturationLevel.STRONG
        case _:
            saturation_enum = SaturationLevel.NONE

    # 設定を作成
    config = PixelArtConfig(
        scale_factor=scale_factor,
        colors=colors,
        filter_type=filter_enum,
        gaussian_sigma=gaussian_sigma,
        erosion_size=erosion_size,
        apply_kmeans=apply_kmeans,
        saturation_level=saturation_enum,
        apply_color_temperature=apply_color_temperature,
        color_temperature_offset=color_temperature_offset,
    )

    # 画像処理を実行
    return await process_image(input_img, config)
