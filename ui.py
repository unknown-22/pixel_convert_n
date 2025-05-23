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

                    colors = gr.Slider( # This is the existing colors slider
                        minimum=2, maximum=32, value=8, step=1, label="色数", visible=True 
                    )

                    apply_kmeans = gr.Checkbox(value=True, label="K-meansで減色する")

                    palette_method = gr.Radio(
                        choices=["K-means", "Custom Palette"],
                        value="K-means",
                        label="Color Reduction Method"
                    )
                    custom_palette_str = gr.Textbox(
                        label="Custom Palette (e.g., #FF0000,#00FF00,#0000FF)",
                        placeholder="Enter comma-separated hex colors",
                        visible=False, # Initially hidden
                        elem_id="custom_palette_textbox"
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
            outputs=[gaussian_sigma, erosion_size],
        )

        # Palette settings visibility control
        def update_palette_settings(selection: str) -> tuple[gr.update, gr.update]:
            if selection == "K-means":
                return gr.update(visible=True), gr.update(visible=False)
            elif selection == "Custom Palette":
                return gr.update(visible=False), gr.update(visible=True)
            return gr.update(visible=True), gr.update(visible=False) # Default fallback

        palette_method.change(
            fn=update_palette_settings,
            inputs=palette_method,
            outputs=[colors, custom_palette_str]
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
                palette_method, 
                custom_palette_str
            ],
            outputs=[output_image, small_image],
        )

    return interface
