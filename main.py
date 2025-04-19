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
) -> tuple[np.ndarray, np.ndarray]:
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
                    image[:, :, i] = morphology.erosion(
                        image[:, :, i],
                        footprint
                    )
            else:
                image = morphology.erosion(image, footprint)
        case _:
            pass
    
    # PILを使用して縮小→拡大（モザイク化）
    pil_img = Image.fromarray((image * 255).astype(np.uint8) if image.dtype == np.float64 else image)
    
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
        pixels = result.reshape(-1, 3) if len(original_shape) == 3 else result.reshape(-1, 1)
        
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
        small_pixels = small_array.reshape(-1, 3) if len(small_shape) == 3 else small_array.reshape(-1, 1)
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
    apply_kmeans: bool
) -> tuple[np.ndarray, np.ndarray]:
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
                    
                    # Gradio 5.xでの表示制御用のスライダーコンテナ
                    gaussian_sigma = gr.Slider(
                        minimum=0.1, 
                        maximum=5.0, 
                        value=1.0, 
                        step=0.1, 
                        label="ガウシアンフィルタの強さ",
                        visible=False  # 初期状態は非表示
                    )
                    
                    erosion_size = gr.Slider(
                        minimum=1, 
                        maximum=5, 
                        value=1, 
                        step=1, 
                        label="エロージョンの強さ",
                        visible=False  # 初期状態は非表示
                    )
                
                convert_btn = gr.Button("変換", variant="primary")
            
            with gr.Column():
                with gr.Row():
                    with gr.Column(scale=3):
                        output_image = gr.Image(label="ドット絵 (拡大後)", type="numpy")
                    with gr.Column(scale=1):
                        small_image = gr.Image(label="縮小画像 (拡大前)", type="numpy")
        
        # フィルタータイプに応じた設定の表示・非表示の制御
        def update_filter_settings(filter_type: str) -> tuple[bool, bool]:
            """
            選択されたフィルタータイプに基づいて、各スライダーの表示状態を更新します
            
            Parameters
            ----------
            filter_type : str
                選択されたフィルタータイプ
                
            Returns
            -------
            tuple[bool, bool]
                (gaussian_sigmaの表示状態, erosion_sizeの表示状態)
            """
            match filter_type:
                case "ガウシアンフィルタ":
                    return True, False
                case "エロージョン":
                    return False, True
                case _:
                    return False, False
        
        filter_type.change(
            fn=update_filter_settings,
            inputs=filter_type,
            outputs=[gaussian_sigma, erosion_size]
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
            outputs=[output_image, small_image]
        )
        
    return interface


async def main():
    """メイン関数"""
    interface = create_ui()
    interface.queue()
    interface.launch(share=False)


if __name__ == "__main__":
    asyncio.run(main())
