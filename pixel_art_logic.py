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


async def process_image(
    image: np.ndarray, config: PixelArtConfig
) -> tuple[np.ndarray, np.ndarray]:
    """
    画像をドット絵に変換する処理を行う

    Parameters
    ----------
    image : np.ndarray
        変換する元の画像
    config : PixelArtConfig
        変換設定パラメータ

    Returns
    -------
    tuple[np.ndarray, np.ndarray]
        (ドット絵画像, 縮小画像)
    """
    # 非同期処理のシミュレーション
    await asyncio.sleep(0.01)

    # 入力画像のサイズを取得
    height, width = image.shape[:2]

    # 色温度調整（前処理として適用）
    if config.apply_color_temperature:
        # 0-1の範囲で正規化されていることを確認
        if image.dtype != np.float64:
            image = image.astype(np.float64) / 255.0
        image = adjust_color_temperature(image, config.color_temperature_offset)

    # 彩度調整（前処理として適用）
    if config.saturation_level != SaturationLevel.NONE:
        # NumPy配列をPIL Imageに変換
        pil_img = Image.fromarray(
            (image * 255).astype(np.uint8) if image.dtype == np.float64 else image
        )
        
        # 彩度調整係数を設定
        match config.saturation_level:
            case SaturationLevel.WEAK:
                saturation_factor = 1.3
            case SaturationLevel.STRONG:
                saturation_factor = 1.8
            case _:
                saturation_factor = 1.0
        
        # 彩度調整を適用
        enhancer = ImageEnhance.Color(pil_img)
        pil_img = enhancer.enhance(saturation_factor)
        
        # NumPy配列に戻す
        image = np.array(pil_img).astype(np.float64) / 255.0

    # 前処理フィルタの適用
    match config.filter_type:
        case FilterType.GAUSSIAN:
            image = gaussian(image, sigma=config.gaussian_sigma, channel_axis=-1)
        case FilterType.EROSION:
            # エロージョン処理（各チャネルに対して適用）
            size = int(config.erosion_size)  # 明示的に整数に変換
            footprint = morphology.footprint_rectangle((size, size))
            if len(image.shape) == 3:
                for i in range(image.shape[2]):
                    image[:, :, i] = morphology.erosion(image[:, :, i], footprint)
            else:
                image = morphology.erosion(image, footprint)
        case _:
            pass

    # PILを使用して縮小→拡大（モザイク化）
    pil_img = Image.fromarray(
        (image * 255).astype(np.uint8) if image.dtype == np.float64 else image
    )

    # 縮小
    new_width = max(1, int(width * config.scale_factor))
    new_height = max(1, int(height * config.scale_factor))
    small_img = pil_img.resize((new_width, new_height), Image.Resampling.LANCZOS)

    # 縮小画像をNumPy配列に変換
    small_array = np.array(small_img)

    # 拡大（ピクセルを大きく）
    pixelated_img = small_img.resize((width, height), Image.Resampling.NEAREST)

    # NumPy配列に戻す
    result = np.array(pixelated_img)

    # K-meansで減色
    if config.apply_kmeans and config.colors > 0:
        # 画像の形状を保存
        original_shape = result.shape

        # 画像をリシェイプ（各ピクセルを1行として扱う）
        pixels = (
            result.reshape(-1, 3) if len(original_shape) == 3 else result.reshape(-1, 1)
        )

        # K-meansクラスタリングを実行
        kmeans = KMeans(n_clusters=config.colors, random_state=42, n_init=10)
        labels = kmeans.fit_predict(pixels)
        centers = kmeans.cluster_centers_

        # 各ピクセルをそのクラスタの中心値に置き換え
        reduced_pixels = centers[labels]

        # 元の形状に戻す
        result = reduced_pixels.reshape(original_shape)

        # 縮小画像にも同じ減色処理を適用
        small_shape = small_array.shape
        small_pixels = (
            small_array.reshape(-1, 3)
            if len(small_shape) == 3
            else small_array.reshape(-1, 1)
        )
        small_labels = kmeans.predict(small_pixels)
        small_reduced = centers[small_labels].reshape(small_shape)
        small_array = small_reduced

    return result.astype(np.uint8), small_array.astype(np.uint8)


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
