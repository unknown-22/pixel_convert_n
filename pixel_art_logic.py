import asyncio
from dataclasses import dataclass
from enum import Enum, auto
from typing import Optional, List

import numpy as np
from PIL import Image
from skimage import morphology
from skimage.filters import gaussian
from sklearn.cluster import KMeans


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
    custom_palette: Optional[List[tuple[int, int, int]]] = None # New field
    palette_method: str = "kmeans"  # New field


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
    # small_array was already converted to NumPy array after resize

    # K-means or Custom Palette for color reduction
    if config.apply_kmeans: # Only proceed if color reduction is enabled
        if config.palette_method == "Custom Palette" and \
           config.custom_palette and \
           len(config.custom_palette) > 0:
            
            # Custom Palette Logic
            original_shape = result.shape
            # Assuming custom palette is for color images, so reshape to (-1, 3)
            # If original image is grayscale, this might implicitly convert it or error
            # For robust grayscale handling with custom color palettes, image would need conversion to color first.
            # Current logic assumes result has 3 channels if a custom palette is applied.
            pixels = result.reshape(-1, 3) 
            
            custom_centers = np.array(config.custom_palette, dtype=np.float64)
            
            kmeans = KMeans(n_clusters=len(custom_centers), n_init=1, random_state=42)
            kmeans.cluster_centers_ = custom_centers
            
            # Ensure KMeans is properly initialized. Fitting on the custom_centers themselves
            # should be sufficient to prepare for predict, if custom_centers is not empty.
            if custom_centers.shape[0] > 0:
                kmeans.fit(custom_centers) # Fit on the defined palette centers.
            
            labels = kmeans.predict(pixels)
            reduced_pixels = custom_centers[labels]
            result = reduced_pixels.reshape(original_shape)

            # Apply to small_array as well
            small_shape = small_array.shape
            small_pixels = small_array.reshape(-1, 3) # Assuming 3 channels
            # No need to re-fit kmeans, it's already configured with custom_centers
            small_labels = kmeans.predict(small_pixels)
            small_reduced = custom_centers[small_labels].reshape(small_shape)
            small_array = small_reduced
            
        elif config.colors > 0: # Fallback to K-means if not custom or custom_palette invalid
            # Existing K-means logic
            original_shape = result.shape
            pixels = (
                result.reshape(-1, 3) if len(original_shape) == 3 else result.reshape(-1, 1)
            )
            kmeans = KMeans(n_clusters=config.colors, random_state=42, n_init=10)
            labels = kmeans.fit_predict(pixels)
            centers = kmeans.cluster_centers_
            reduced_pixels = centers[labels]
            result = reduced_pixels.reshape(original_shape)

            small_shape = small_array.shape
            small_pixels = (
                small_array.reshape(-1, 3)
                if len(small_shape) == 3
                else small_array.reshape(-1, 1)
            )
            # Use predict here on existing kmeans model, not fit_predict
            small_labels = kmeans.predict(small_pixels) 
            small_reduced = centers[small_labels].reshape(small_shape)
            small_array = small_reduced
        # If config.colors <= 0 and not using a valid custom palette, no color reduction happens here.

    return result.astype(np.uint8), small_array.astype(np.uint8)


async def pixel_art_converter(
    input_img: np.ndarray,
    scale_factor: float,
    colors: int,
    filter_type: str,
    gaussian_sigma: float,
    erosion_size: int,
    apply_kmeans: bool,
    palette_method: str, # New parameter
    custom_palette_str: Optional[str] # New parameter
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
    palette_method : str
        パレット選択方法 ("K-means" または "Custom Palette")
    custom_palette_str : Optional[str]
        カスタムパレットのカンマ区切りHEX文字列 (例: "#FF0000,#00FF00")

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

    # Parse custom_palette_str
    parsed_custom_palette: Optional[List[tuple[int, int, int]]] = None
    if palette_method == "Custom Palette" and custom_palette_str:
        temp_palette = []
        hex_colors = custom_palette_str.split(',')
        valid_palette = True
        for hex_color in hex_colors:
            hex_color = hex_color.strip()
            if hex_color.startswith('#') and len(hex_color) == 7:
                try:
                    r = int(hex_color[1:3], 16)
                    g = int(hex_color[3:5], 16)
                    b = int(hex_color[5:7], 16)
                    temp_palette.append((r, g, b))
                except ValueError:
                    valid_palette = False # Invalid hex char
                    break 
            else:
                valid_palette = False # Invalid format
                break
        if valid_palette and temp_palette: # Ensure temp_palette is not empty
            parsed_custom_palette = temp_palette
        # If not valid_palette or temp_palette is empty, parsed_custom_palette remains None

    # 設定を作成
    config = PixelArtConfig(
        scale_factor=scale_factor,
        colors=colors,
        filter_type=filter_enum,
        gaussian_sigma=gaussian_sigma,
        erosion_size=erosion_size,
        apply_kmeans=apply_kmeans,
        palette_method=palette_method,
        custom_palette=parsed_custom_palette
    )

    # 画像処理を実行
    return await process_image(input_img, config)
