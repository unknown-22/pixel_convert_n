# Pixel Converter Next

Gradioでドット絵化をするPythonプログラムです。

仕様
- GradioでWebUIを表示
- 入力された画像をドット絵化(Pixel Art化)
- 処理内容
    - 縮小→拡大（モザイク化）
    - ピクセルの色をk色にkmeansで分類しそれぞれの平均値を求め、色を置き換え減色します
    - 透過PNGの境界色を保ち、アルファを二値化して変換します
- オプション処理
    - ガウシアンフィルタ
    - バイラテラルフィルタ
    - エロージョン
    - 減色(kmeans)
    - 規則的ディザリング(Bayer 4×4)
    - 彩度調節(なし/弱/強)
    - 色温度調節(-35 ~ +35)

TODO
- 輪郭線の膨張(なし/弱/強)
- コントラストを上げる(なし/弱/強)

### Web UIの起動

```bash
. .venv/bin/activate
python main.py
```

### バッチ変換CLI

Gradio を使わずに、ディレクトリ内の PNG を再帰的にドット絵化して `*_converted.png` として保存します。`*_converted.png` 自体は入力対象になりません。出力が既にある場合はデフォルトでスキップし、`--overwrite` を指定した場合だけ上書きします。

実行例:
```bash
python batch_pixel_art_converter.py \
  /path/to/directory \
  --scale-factor 0.15 \
  --colors 32 \
  --apply-kmeans \
  --saturation-level weak \
  --filter-type none
```

主なオプション（未指定は PixelArtConfig のデフォルト）
- `--scale-factor` : 縮小率（例 0.15）
- `--colors` : 減色後の色数（1〜256）
- `--filter-type` : `none` / `gaussian` / `bilateral` / `erosion`
- `--gaussian-sigma` : ガウシアンフィルタのシグマ
- `--erosion-size` : エロージョンのカーネルサイズ
- `--apply-kmeans` / `--no-apply-kmeans`
- `--saturation-level` : `none` / `weak` / `strong`
- `--apply-color-temperature` / `--no-apply-color-temperature`
- `--color-temperature-offset` : 色温度オフセット（-35〜35）
- `--dithering` : `none` / `ordered`（Bayer 4×4）
- `--dithering-strength` : ディザリング強度（推奨 0.08〜0.15）
- `--overwrite` : 既存の変換結果を上書き

処理後に成功・スキップ・失敗件数を表示します。1件でも失敗した場合の終了コードは `1` です。

### 変換処理と画質の設定

処理順序は「色調整 → エロージョン（任意）→ 平滑化（任意）→ 縮小 → 可視画素のみでK-meansパレット生成 → ディザリングを含む減色 → 最近傍拡大」です。
WebUIでは拡大画像と縮小画像を表示します。CLIは従来どおり縮小画像を保存します。

- 縮小方法: `nearest`（既定）または `lanczos`。WebUIの「縮小方法」、CLIの `--resize-method` で選択します。
- 平滑化: なし／ガウシアン／バイラテラル。バイラテラルは輪郭を保ちながら細かな色変化を抑えます。
- エロージョン: 平滑化と独立して有効化でき、先に適用されます。サイズ1は変化なし、3からの比較を推奨します。
- バイラテラル: scikit-imageの最適化済み実装と15×15近傍を使用します。乗算済みRGBAを処理することで透明境界の色にじみを防ぎます。
- 透過: リサイズ時はRGBとアルファを乗算して処理し、元の色を復元した後にアルファを閾値0.5で二値化します。半透明の階調は保持しません。透明画素は減色の学習から除外します。
- ディザリング: 4×4 Bayer行列で縮小画像の明度を規則的に変化させ、K-meansパレット内の色だけで中間色を表現します。K-meansが無効な場合は適用されません。
- K-means: 可視画素が10万を超える場合は、再現可能な等間サンプリングでパレットを学習します。パレットへの色割り当ては全画素に対して行います。
- 色温度: `0` は完全な無変換です。`-35`〜`35` の範囲を受け付けます。

```bash
. .venv/bin/activate
python batch_pixel_art_converter.py /path/to/directory \
  --scale-factor 0.25 --colors 16 --resize-method nearest \
  --filter-type bilateral --bilateral-sigma-color 0.1 \
  --bilateral-sigma-spatial 3 --apply-erosion --erosion-size 3
```

既存の `--filter-type erosion` も引き続き利用できます。平滑化と併用する場合は `--apply-erosion` を指定してください。

### 検証

```bash
. .venv/bin/activate
ruff check
ty check
pytest
```
