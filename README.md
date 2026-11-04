# EmbeddingGemma 2 マルチモーダル検索デモ（日本語 / Google Colab）

[![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/tatsuya-fukuoka/EmbeddingGemma2-demo/blob/main/EmbeddingGemma2_Multimodal_Colab_JA_CUDA_Fix.ipynb)

Google の埋め込みモデル **[EmbeddingGemma 2](https://huggingface.co/google/embeddinggemma-2)** を使い、**日本語テキスト・画像・音声・動画**をまたいだベクトル検索を Google Colab 上で試せるデモノートブックです。

> [!NOTE]
> EmbeddingGemma 2 は文章を生成するモデルではありません。入力を **768 次元のベクトル**に変換するモデルです。このデモでは、ベクトル同士の cosine 類似度で検索します。

---

## このノートブックでできること

| # | 内容 | 入力 → 検索対象 |
|---|------|----------------|
| 1 | 日本語の意味検索 | テキスト → テキスト |
| 2 | 日本語の質問から写真を探す | テキスト → 画像 |
| 3 | 次元数を変えたときの検索順位の比較（MRL） | 768 / 512 / 256 / 128 次元 |
| 4 | 自分の画像をアップロードして検索 | テキスト → 画像 |
| 5 | 音声・動画との照合（オプション） | 音声 / 動画 → テキスト |
| 6 | 障害報告書の意味検索 | Incident Intelligence の入口 |

---

## クイックスタート

1. 上の **Open In Colab** バッジをクリックします。
2. Colab で **[ランタイム] → [ランタイムのタイプを変更] → [T4 GPU]** を選びます（使えるなら L4 GPU）。
3. 上のセルから順に実行します。初回実行時にモデルがダウンロードされます。

- API キーは通常不要です。モデルにアクセス制限がかかっている場合のみ、Hugging Face の認証を追加してください。
- CPU でも動きますが、かなり遅くなります。

---

## 動作環境

| 項目 | 内容 |
|------|------|
| 実行環境 | Google Colab |
| GPU | T4（動作確認済み）/ L4 推奨 |
| モデル | `google/embeddinggemma-2` |
| 主要ライブラリ | `sentence-transformers`, `transformers`, `torch`, `torchcodec`（CPU 版）, `pillow`, `pandas`, `matplotlib`, `soundfile` |
| 検索基盤 | NumPy（ベクトル DB は不要） |

### モデル構成

| 設定 | 読み込むエンコーダ | パラメータ数 |
|------|------------------|-------------|
| `ENABLE_AUDIO = False`（デフォルト） | テキスト + 画像 + 動画 | 約 440M（実測 438,760,448） |
| `ENABLE_AUDIO = True` | 上記 + 音声 | 約 740M |

GPU メモリを節約するため、デフォルトでは音声エンコーダを読み込みません。

### 精度（dtype）

GPU が bfloat16 に対応していれば `bfloat16`、対応していなければ `float32` を自動で選びます。
**`float16` は使わないでください**（`NaN` の原因になります）。

---

## CUDA 13 / TorchCodec エラー対策について

最近の Colab では、CUDA 版の `torchcodec` を読み込むと `libnvrtc.so.13` が見つからずエラーになることがあります。このノートブックでは次の方法で回避しています。

- Colab に入っている **PyTorch は変更しない**
- `torchcodec` だけを、PyTorch のバージョンに合った **CPU 版**に入れ替える（`--force-reinstall --no-deps`）
- EmbeddingGemma 2 の推論そのものは **GPU のまま**動く

| PyTorch | TorchCodec（CPU 版） |
|---------|---------------------|
| 2.7 | 0.5.0 |
| 2.8 | 0.7.0 |
| 2.9 | 0.9.0 |
| 2.10 | 0.10.0 |
| 2.11 | 0.11.1 |
| 2.12 以降 | 0.17.0 |

> [!TIP]
> 同じランタイムですでにエラーが出ている場合は、インストールセルを実行してから **[ランタイム] → [セッションを再起動]** し、上から順に実行し直してください。

---

## ノートブックの構成

| セクション | 内容 |
|-----------|------|
| 1. ライブラリのインストールと CUDA 互換性の修正 | パッケージの追加、CPU 版 TorchCodec の導入、読み込みチェック |
| 2. 実行環境の確認・モデル読み込み | GPU と dtype の判定、`SentenceTransformer` でモデルを読み込む |
| 3. 日本語テキストの意味検索 | 障害報告を題材にした Text → Text 検索 |
| 4. サンプル画像のダウンロード | Gemma Cookbook と Hugging Face のデモ画像 4 枚 |
| 5. 日本語 → 画像のクロスモーダル検索 | 日本語の質問から画像を検索 |
| 6. 次元数の比較（MRL） | 768 / 512 / 256 / 128 次元で順位を比べる |
| 7. 自分の画像をアップロードして検索（任意） | Colab のアップロード機能で画像を追加 |
| 8. 音声 → テキスト検索（任意） | `ENABLE_AUDIO=True` のときだけ実行 |
| 9. 動画 → テキスト検索（任意） | `RUN_VIDEO_DEMO=True` のときだけ実行（最大 8 フレーム） |
| 10. 結果の読み方と次のステップ | 評価の観点、Incident Intelligence への発展案 |

---

## 使い方のポイント

### タスク接頭辞（プロンプト）の使い分け

検索の質問側と、検索される文書側で、別々のプロンプトを付けます。

```python
q_vec = model.encode(query, prompt_name="SearchQuery", normalize_embeddings=True)
doc_vecs = model.encode(docs, prompt_name="Document", normalize_embeddings=True)
scores = doc_vecs @ q_vec  # 正規化済みなので内積 = cosine 類似度
```

### 画像・音声・動画の入力形式

画像・音声・動画は dict で渡します。プロンプト（`SearchQuery`）はテキストにだけ付けます。

```python
model.encode([{"image": "path/to/image.jpg"}], normalize_embeddings=True)
model.encode({"audio": "path/to/audio.wav"}, normalize_embeddings=True)
model.encode(
    {"video": "path/to/video.mp4"},
    normalize_embeddings=True,
    processing_kwargs={"video": {"max_frames": 8, "fps": 1,
                                 "overflow_strategy": "uniform", "add_timestamps": True}},
)
```

### MRL（Matryoshka Representation Learning）

768 次元のベクトルの先頭だけを切り出せば、低い次元でも使えます。**切り出した後は必ずもう一度 L2 正規化してください。**

```python
def truncate_and_normalize(x, dim):
    arr = np.asarray(x, dtype=np.float32)[..., :dim]
    return arr / np.maximum(np.linalg.norm(arr, axis=-1, keepdims=True), 1e-12)
```

次元を減らして節約できるのは、主に**ベクトルの保存と検索のコスト**です。モデルの推論コストは変わりません。

---

## 実行結果の例（Colab T4）

### テキスト → テキスト

クエリ：「回転機器が変な音を出してから止まった事例は？」

| 順位 | 類似度 | 文書 |
|-----|-------|------|
| 1 | 0.8184 | モーターから異音が発生した後、設備が停止した。ベアリング摩耗が原因で交換した。 |
| 2 | 0.7629 | 冷却ファンの故障で温度が上昇し、過熱保護により装置が自動停止した。 |
| 3 | 0.6970 | 日次点検で異常なし。モーター温度と振動は基準範囲内。 |
| 4 | 0.6577 | カメラのレンズが汚れたことで外観検査の誤検知が増え、清掃して解消した。 |
| 5 | 0.6514 | ネットワークのパケット損失が増加し、PLCの通信が断続的になった。ケーブルを交換した。 |

「回転機器」「変な音」という言い方でも、「モーター」「異音」と書かれた文書が 1 位になっています。

### テキスト → 画像

| クエリ | 1 位 | 類似度 |
|-------|------|-------|
| 海にかかる赤い大きな吊り橋の写真 | Golden Gate Bridge | 0.7322 |
| 調理器具が置かれたキッチンのイラスト | Kitchen illustration | 0.7561 |
| 青い車が道路に駐車している | Car | 0.6473 |

### MRL（次元数ごとの比較）

クエリ：「海にかかる赤い大きな吊り橋の写真」

| 次元数 | 容量（768 次元比） | 1 位 | 1 位スコア | 上位 3 件 |
|-------|-----------------|------|-----------|----------|
| 768 | 100.0% | Golden Gate Bridge | 0.7324 | Golden Gate Bridge, Car, Kitchen illustration |
| 512 | 66.7% | Golden Gate Bridge | 0.7302 | Golden Gate Bridge, Car, Kitchen illustration |
| 256 | 33.3% | Golden Gate Bridge | 0.7539 | Golden Gate Bridge, Bee, Car |
| 128 | 16.7% | Golden Gate Bridge | 0.8419 | Golden Gate Bridge, Kitchen illustration, Car |

128 次元でも 1 位は変わりません。2 位以下の順位は次元数によって入れ替わります。

> [!IMPORTANT]
> スコアは **cosine 類似度**です。確率ではありません。モダリティ（テキスト・画像・音声など）によってスコアの分布が違うことがあるので、しきい値を全モダリティで共通にしないでください。

---

## 発展：Multimodal Incident Intelligence

障害対応のナレッジ検索に発展させる場合の流れです。

1. 障害報告書を `Document` プロンプトで埋め込む
2. 現場の写真・異音・動画も、それぞれそのまま Embedding にする
3. モダリティごとの検索結果をまとめ、同じ `incident_id` の事例として集約する
4. テキストのみ / 画像あり / 音声あり / 時系列あり の条件で、**Recall@K・MRR・推論時間**を比べる
5. 原因の推定は別の LLM / VLM に任せる。Embedding の類似度を「原因である確率」として扱わない

データが増えたら、NumPy での検索から **Qdrant** などのベクトル DB に移行できます。

---

## トラブルシューティング

| 症状 | 対処 |
|------|------|
| `libnvrtc.so.13` 関連の TorchCodec 読み込みエラー | インストールセルを実行して CPU 版 TorchCodec を入れ、ランタイムを再起動する |
| モデルが読み込めない | インストールセルをもう一度実行し、必要ならランタイムを再起動する |
| `RuntimeError: CUDA out of memory` | `ENABLE_AUDIO=False` にする / 画像の枚数や動画のフレーム数を減らす |
| 結果に `NaN` が出る | `float16` を使っていないか確認する |
| サンプル画像のダウンロードに失敗する | セクション 7 で自分の画像をアップロードして試す |
| 音声のセルがスキップされる | セクション 2 で `ENABLE_AUDIO=True` にし、モデル読み込み以降のセルを再実行する |

---

## 参考リンク

- [EmbeddingGemma 2（Hugging Face）](https://huggingface.co/google/embeddinggemma-2)
- [公式ガイド：マルチモーダル EmbeddingGemma × Sentence Transformers](https://ai.google.dev/gemma/docs/embeddinggemma/multimodal-embeddinggemma-with-sentence-transformers)
- [公式ガイド：EmbeddingGemma のテキスト推論](https://ai.google.dev/gemma/docs/embeddinggemma/inference-embeddinggemma-with-sentence-transformers)
- [Gemma Cookbook](https://github.com/google-gemma/cookbook)

## ライセンス・利用上の注意

- モデルは [Gemma の利用規約](https://ai.google.dev/gemma/terms) に従って利用してください。
- サンプル画像・音声・動画の著作権は、それぞれの配布元に帰属します。
