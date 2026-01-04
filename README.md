# Pixel Converter Next

Gradioでドット絵化をするPythonプログラムです。

仕様
- GradioでWebUIを表示
- 入力された画像をドット絵化(Pixel Art化)
- 処理内容
    - 縮小→拡大（モザイク化）
    - ピクセルの色をk色にkmeansで分類しそれぞれの平均値を求め、色を置き換え減色します
    - 透過PNGのアルファ値を保持したまま変換します
- オプション処理
    - ガウシアンフィルタ
    - エロージョン
    - 減色(kmeans)
    - 彩度調節(なし/弱/強)
    - 色温度調節(-35 ~ +35)

TODO
- 輪郭線の膨張(なし/弱/強)
- コントラストを上げる(なし/弱/強)
- ディザリング

ライブラリ
- Pillow
- scikit‑image
- scikit‑learn
- gradio

### バッチ変換CLI

Gradio を使わずに、ディレクトリ内の PNG を再帰的にドット絵化して `*_converted.png` として保存するスクリプトを追加しました。

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
- `--colors` : 減色後の色数
- `--filter-type` : `none` / `gaussian` / `erosion`
- `--gaussian-sigma` : ガウシアンフィルタのシグマ
- `--erosion-size` : エロージョンのカーネルサイズ
- `--apply-kmeans` / `--no-apply-kmeans`
- `--saturation-level` : `none` / `weak` / `strong`
- `--apply-color-temperature` / `--no-apply-color-temperature`
- `--color-temperature-offset` : 色温度オフセット（推奨 -35〜35）

### 📝 これまで決まった仕様まとめ

- 📦 **使用ライブラリ**  
  - Pillow : 画像の読み込み／保存  
  - NumPy : 配列・数値演算  
  - scikit‑image : リサイズ（モザイク化）、平滑化フィルタ、形態学処理（erode など）  
  - scikit‑learn : `KMeans` で減色（k 色クラスタリング）  
  - Gradio : Web UI（インターフェース）

- 🔧 **入力パラメータ**  
  - `path_in` / `path_out` : 入力・出力ファイルパス  
  - `scale` : モザイク化用の縮小率（例 0.1 → 1/10 サイズ）  
  - `k` : 残す色数（クラスタ数）  
  - `smooth` (真偽) : ガウシアンなどで平滑化するか  
  - `erode` (真偽) : 形態学的エロージョンをかけるか  

- 🖼 **処理フロー**  
  1. 画像読み込み（RGB → `uint8`）  
  2. **縮小→拡大（Nearest）** でモザイク化  
  3. 必要に応じて  
     - 平滑化 : `skimage.filters.gaussian` など  
     - エロージョン : `skimage.morphology.erosion`  
  4. 画素を k 色に **`KMeans` でクラスタリング** → 代表色に置換  
  5. `uint8` に戻し、Pillow で保存  

- ⚙️ **実装時の注意**  
  - scikit‑image 関数は `float32 [0–1]` を返すことが多い → `img_as_ubyte()` で戻す  
  - リサイズは `order=0`（最近傍補間）かつ `anti_aliasing=False`  
  - `KMeans(n_init="auto")` を使うと scikit‑learn 1.4+ で警告なし  
  - 処理速度をさらに上げたい／機能を増やしたい場合のみ OpenCV 併用を検討  
