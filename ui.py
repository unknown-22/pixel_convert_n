import gradio as gr

from pixel_art_logic import pixel_art_converter


def create_ui() -> gr.Blocks:
    """
    Gradioを使用したUIを作成する

    Returns
    -------
    gr.Blocks
        Gradioインターフェース
    """
    with gr.Blocks(title="ドット絵変換ツール") as interface:
        gr.Markdown("# 🎮 ドット絵変換ツール")
        gr.Markdown(
            "画像をドット絵（ピクセルアート）に変換します。パラメータを調整して好みの結果を得ましょう。"
        )

        with gr.Row():
            with gr.Column():
                input_image = gr.Image(
                    label="元の画像",
                    type="numpy",
                    sources="upload",
                    format="png",
                    image_mode="RGBA",  # 透過情報を保持
                    elem_classes="input-image",
                )

                with gr.Group():
                    gr.Markdown("## 基本設定")
                    scale_factor = gr.Slider(
                        minimum=0.05,
                        maximum=0.5,
                        value=0.2,
                        step=0.05,
                        label="縮小率 (小さいほどドットが大きい)",
                    )

                    colors = gr.Slider(
                        minimum=1, maximum=32, value=8, step=1, label="色数"
                    )

                    apply_kmeans = gr.Checkbox(value=True, label="K-meansで減色する")

                with gr.Group():
                    gr.Markdown("## 画像調整")
                    saturation_level = gr.Radio(
                        choices=["なし", "弱", "強"],
                        value="なし",
                        label="彩度調整",
                    )

                    apply_color_temperature = gr.Checkbox(
                        value=False, 
                        label="色温度調整"
                    )

                    color_temperature_offset = gr.Slider(
                        minimum=-35,
                        maximum=35,
                        value=0,
                        step=1,
                        label="色温度調整 (±1で±100K、0=標準6500K)",
                        info="プラス値で暖色系（赤み）、マイナス値で寒色系（青み）",
                        visible=False,  # 初期状態は非表示
                    )

                with gr.Group():
                    gr.Markdown("## フィルター設定")
                    filter_type = gr.Radio(
                        choices=["なし", "ガウシアンフィルタ", "エロージョン"],
                        value="なし",
                        label="前処理フィルター",
                    )

                    # Gradio 5.xでの表示制御用のスライダーコンテナ
                    gaussian_sigma = gr.Slider(
                        minimum=0.1,
                        maximum=5.0,
                        value=1.0,
                        step=0.1,
                        label="ガウシアンフィルタの強さ",
                        visible=False,  # 初期状態は非表示
                    )

                    erosion_size = gr.Slider(
                        minimum=1,
                        maximum=5,
                        value=1,
                        step=1,
                        label="エロージョンの強さ",
                        visible=False,  # 初期状態は非表示
                    )

                convert_btn = gr.Button("変換", variant="primary")

            with gr.Column():
                with gr.Row():
                    with gr.Column(scale=3):
                        output_image = gr.Image(
                            label="ドット絵 (拡大後)",
                            type="numpy",
                            format="png",
                            image_mode="RGBA",  # 透過を表示
                        )
                    with gr.Column(scale=1):
                        small_image = gr.Image(
                            label="縮小画像 (拡大前)",
                            type="numpy",
                            format="png",
                            image_mode="RGBA",  # 透過を表示
                        )

        # フィルタータイプに応じた設定の表示・非表示の制御
        def update_filter_settings(filter_type: str) -> tuple[gr.update, gr.update]:
            """
            選択されたフィルタータイプに基づいて、各スライダーの表示状態を更新します

            Parameters
            ----------
            filter_type : str
                選択されたフィルタータイプ

            Returns
            -------
            tuple[gr.update, gr.update]
                (gaussian_sigmaの表示状態, erosion_sizeの表示状態)
            """
            match filter_type:
                case "ガウシアンフィルタ":
                    return gr.update(visible=True), gr.update(visible=False)
                case "エロージョン":
                    return gr.update(visible=False), gr.update(visible=True)
                case _:
                    return gr.update(visible=False), gr.update(visible=False)

        # 色温度調整の表示・非表示制御
        def update_color_temperature_visibility(apply_temp: bool) -> gr.update:
            """
            色温度調整チェックボックスの状態に基づいて、色温度スライダーの表示状態を更新します

            Parameters
            ----------
            apply_temp : bool
                色温度調整を適用するかどうか

            Returns
            -------
            gr.update
                色温度スライダーの表示状態更新オブジェクト
            """
            return gr.update(visible=apply_temp)

        filter_type.change(
            fn=update_filter_settings,
            inputs=filter_type,
            outputs=[gaussian_sigma, erosion_size],
        )

        apply_color_temperature.change(
            fn=update_color_temperature_visibility,
            inputs=apply_color_temperature,
            outputs=color_temperature_offset,
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
                apply_kmeans,
                saturation_level,
                apply_color_temperature,
                color_temperature_offset,
            ],
            outputs=[output_image, small_image],
        )

    return interface
