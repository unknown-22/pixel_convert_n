"""
ドット絵変換アプリケーション

画像をドット絵（ピクセルアート）に変換するGradioベースのWebアプリケーション。
縮小・拡大処理によるモザイク化、k-meansクラスタリングによる減色などの
処理オプションを提供します。
"""

import asyncio
from dataclasses import dataclass, field
from enum import Enum, auto
from pathlib import Path
from typing import Any, Callable

import gradio as gr
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


async def process_image(
    image: np.ndarray,
    config: PixelArtConfig
) -> np.ndarray:
    """
    画像をドット絵に変換する

    Parameters
    ----------
    image : np.ndarray
        入力画像
    config : PixelArtConfig
        変換設定

    Returns
    -------
    np.ndarray
        変換された画像
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
            if len(image.shape) == 3:
                for i in range(image.shape[2]):
                    image[:, :, i] = morphology.erosion(
                        image[:, :, i],
                        morphology.square(config.erosion_size)
                    )
            else:
                image = morphology.erosion(image, morphology.square(config.erosion_size))
        case _:
            pass
    
    # PILを使用して縮小→拡大（モザイク化）
    pil_img = Image.fromarray((image * 255).astype(np.uint8) if image.dtype == np.float64 else image)
    
    # 縮小
    new_width = max(1, int(width * config.scale_factor))
    new_height = max(1, int(height * config.scale_factor))
    small_img = pil_img.resize((new_width, new_height), Image.Resampling.LANCZOS)
    
    # 拡大（ピクセルを大きく）
    pixelated_img = small_img.resize((width, height), Image.Resampling.NEAREST)
    
    # NumPy配列に戻す
    result = np.array(pixelated_img)
    
    # K-meansで減色
    if config.apply_kmeans and config.colors > 0:
        # 画像の形状を保存
        original_shape = result.shape
        
        # 画像をリシェイプ（各ピクセルを1行として扱う）
        pixels = result.reshape(-1, 3) if len(original_shape) == 3 else result.reshape(-1, 1)
        
        # K-meansクラスタリングを実行
        kmeans = KMeans(n_clusters=config.colors, random_state=42, n_init=10)
        labels = kmeans.fit_predict(pixels)
        centers = kmeans.cluster_centers_
        
        # 各ピクセルをそのクラスタの中心値に置き換え
        reduced_pixels = centers[labels]
        
        # 元の形状に戻す
        result = reduced_pixels.reshape(original_shape)
    
    return result.astype(np.uint8)


async def pixel_art_converter(
    input_img: np.ndarray,
    scale_factor: float,
    colors: int,
    filter_type: str,
    gaussian_sigma: float,
    erosion_size: int,
    apply_kmeans: bool
) -> np.ndarray:
    """
    Gradioインターフェース用のラッパー関数

    Parameters
    ----------
    input_img : np.ndarray
        入力画像
    scale_factor : float
        縮小率
    colors : int
        色数
    filter_type : str
        フィルタータイプ
    gaussian_sigma : float
        ガウシアンフィルタのシグマ値
    erosion_size : int
        エロージョンのカーネルサイズ
    apply_kmeans : bool
        k-meansによる減色を適用するかどうか

    Returns
    -------
    np.ndarray
        変換された画像
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
    
    # 設定を作成
    config = PixelArtConfig(
        scale_factor=scale_factor,
        colors=colors,
        filter_type=filter_enum,
        gaussian_sigma=gaussian_sigma,
        erosion_size=erosion_size,
        apply_kmeans=apply_kmeans
    )
    
    # 画像処理を実行
    return await process_image(input_img, config)


def create_ui() -> gr.Blocks:
    """
    Gradio UIを作成する

    Returns
    -------
    gr.Blocks
        Gradioのブロックインターフェース
    """
    with gr.Blocks(title="ドット絵変換ツール") as interface:
        gr.Markdown("# 🎮 ドット絵変換ツール")
        gr.Markdown("画像をドット絵（ピクセルアート）に変換します。パラメータを調整して好みの結果を得ましょう。")
        
        with gr.Row():
            with gr.Column():
                input_image = gr.Image(
                    label="元の画像",
                    type="numpy",
                    sources="upload",
                    elem_classes="input-image"
                )
                
                with gr.Group():
                    gr.Markdown("## 基本設定")
                    scale_factor = gr.Slider(
                        minimum=0.05, 
                        maximum=0.5, 
                        value=0.2, 
                        step=0.05, 
                        label="縮小率 (小さいほどドットが大きい)"
                    )
                    
                    colors = gr.Slider(
                        minimum=2, 
                        maximum=32, 
                        value=8, 
                        step=1, 
                        label="色数"
                    )
                    
                    apply_kmeans = gr.Checkbox(
                        value=True, 
                        label="K-meansで減色する"
                    )
                
                with gr.Group():
                    gr.Markdown("## フィルター設定")
                    filter_type = gr.Radio(
                        choices=["なし", "ガウシアンフィルタ", "エロージョン"],
                        value="なし",
                        label="前処理フィルター"
                    )
                    
                    with gr.Group(visible=False) as gaussian_group:
                        gaussian_sigma = gr.Slider(
                            minimum=0.1, 
                            maximum=5.0, 
                            value=1.0, 
                            step=0.1, 
                            label="ガウシアンフィルタの強さ"
                        )
                    
                    with gr.Group(visible=False) as erosion_group:
                        erosion_size = gr.Slider(
                            minimum=1, 
                            maximum=5, 
                            value=1, 
                            step=1, 
                            label="エロージョンの強さ"
                        )
                
                convert_btn = gr.Button("変換", variant="primary")
            
            with gr.Column():
                output_image = gr.Image(label="変換結果", type="numpy")
        
        # フィルタータイプに応じた設定の表示・非表示の制御
        def update_filter_settings(filter_type):
            match filter_type:
                case "ガウシアンフィルタ":
                    return {
                        gaussian_group: gr.Group.update(visible=True),
                        erosion_group: gr.Group.update(visible=False)
                    }
                case "エロージョン":
                    return {
                        gaussian_group: gr.Group.update(visible=False),
                        erosion_group: gr.Group.update(visible=True)
                    }
                case _:
                    return {
                        gaussian_group: gr.Group.update(visible=False),
                        erosion_group: gr.Group.update(visible=False)
                    }
        
        filter_type.change(
            fn=update_filter_settings,
            inputs=filter_type,
            outputs=[gaussian_group, erosion_group]
        )
        
        # 変換ボタンのイベントハンドラ
        convert_btn.click(
            fn=pixel_art_converter,
            inputs=[
                input_image,
                scale_factor,
                colors,
                filter_type,
                gaussian_sigma,
                erosion_size,
                apply_kmeans
            ],
            outputs=output_image
        )
        
    return interface


async def main():
    """メイン関数"""
    interface = create_ui()
    interface.queue()
    interface.launch(share=False)


if __name__ == "__main__":
    asyncio.run(main())
