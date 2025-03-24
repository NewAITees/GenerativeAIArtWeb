# 開発環境セットアップガイド

このドキュメントでは、GenerativeAIArtWebプロジェクトの開発環境を構築するための手順を説明します。特に非同期処理機能を使用したGradioインターフェースの開発に焦点を当てています。

## 前提条件

以下のソフトウェアがインストールされていることを確認してください：

- Python 3.11以上
- Poetry (依存関係管理用)
- Git
- NVIDIAドライバとCUDA（GPUを使用する場合）

## 環境セットアップ手順

### 1. リポジトリのクローン

```bash
# リポジトリをクローン
git clone https://github.com/yourusername/GenerativeAIArtWeb.git
cd GenerativeAIArtWeb
```

### 2. 依存パッケージのインストール

```bash
# Poetryがインストールされていない場合
curl -sSL https://install.python-poetry.org | python3 -

# 依存パッケージのインストール
poetry install

# オプション: CUDA サポートを有効にする場合
poetry install -E cuda
```

### 3. モデルファイルの準備

必要なモデルファイルを `models` ディレクトリに配置します：

- `clip_g.safetensors` (OpenCLIP bigG、SDXLと同じ)
- `clip_l.safetensors` (OpenAI CLIP-L、SDXLと同じ)
- `t5xxl.safetensors` または `t5xxl_fp16.safetensors` (Google T5-v1.1-XXL)
- `sd3.5_large.safetensors` または別のSD3系モデルファイル
- オプション: `sd3_vae.safetensors` (VAEが別ファイルの場合)
- `llm_model.safetensors` (プロンプト生成用LLM)

### 4. オプション：Ollamaのインストールと設定

LLMベースのプロンプト生成機能を使用するには、Ollamaをインストールして設定します：

```bash
# Linux/macOSの場合
curl -fsSL https://ollama.com/install.sh | sh

# Windowsの場合は公式サイトからインストーラーをダウンロード
# https://ollama.com/download/windows

# llama3モデルのダウンロード
ollama pull llama3
```

### 5. 環境変数の設定

`.env` ファイルを作成して以下の環境変数を設定します：

```
# モデルパス
MODEL_FOLDER=models
# 出力先ディレクトリ
OUTPUT_DIR=outputs
# Ollamaホスト（デフォルトはlocalhost）
OLLAMA_HOST=http://localhost:11434
```

## 開発サーバーの起動

### 通常モード

```bash
# 通常モードでアプリを起動
poetry run python src/web/app.py
```

### デモモード

```bash
# デモモードでアプリを起動（実際のモデルを使わない）
poetry run python src/web/demo.py
```

### 開発オプション

```bash
# デバッグモードで起動
poetry run python src/web/app.py --debug

# 公開リンクを生成して起動
poetry run python src/web/app.py --share
```

## フォルダ構造

主要なディレクトリとファイルの説明：

```
GenerativeAIArtWeb/
├── models/              # モデルファイル保存ディレクトリ
├── outputs/             # 生成画像の出力ディレクトリ
├── settings/            # 設定ファイル保存ディレクトリ
├── src/                 # ソースコード
│   ├── generator/       # 画像生成エンジン
│   │   ├── sd3_inf.py   # SD3インフェレンサー
│   │   ├── sd3_impls.py # SD3実装
│   │   └── async_wrapper.py # 非同期ラッパー
│   ├── web/             # Webインターフェース
│   │   ├── app.py       # Gradioアプリケーション
│   │   └── demo.py      # デモモード
│   ├── prompt/          # プロンプト生成
│   │   ├── llm_generator.py # LLMベースの生成
│   │   └── json_builder.py  # JSONベースの生成
│   ├── utils/           # ユーティリティ
│   │   ├── upscaler.py    # 解像度向上
│   │   ├── watermark.py   # ウォーターマーク
│   │   ├── file_manager.py # ファイル管理
│   │   └── settings.py    # 設定管理
│   └── models/          # Pydanticモデル
└── test/                # テストコード
```

## 非同期Gradioの開発

### 基本的な非同期イベントハンドラの例

```python
import gradio as gr
import asyncio

async def async_process(text, progress=gr.Progress()):
    # 進捗表示の初期化
    progress(0, desc="処理を開始しています...")
    
    # 処理ステップのシミュレーション
    total_steps = 10
    for i in range(total_steps):
        # 時間のかかる処理
        await asyncio.sleep(0.5)
        
        # 進捗を更新
        progress((i+1)/total_steps, desc=f"ステップ {i+1}/{total_steps}")
    
    return f"処理結果: {text}"

# Gradioインターフェース
with gr.Blocks() as demo:
    input_text = gr.Textbox(label="入力")
    output_text = gr.Textbox(label="出力")
    
    process_btn = gr.Button("処理開始")
    
    # 非同期関数をイベントハンドラとして登録
    process_btn.click(
        fn=async_process,
        inputs=input_text,
        outputs=output_text
    )

demo.launch()
```

### ストリーミング出力の例

```python
import gradio as gr
import asyncio
import time

async def stream_results(query):
    # 初期出力
    yield "処理を開始します..."
    
    # ストリーミング出力のシミュレーション
    for i in range(5):
        await asyncio.sleep(1)
        yield f"処理ステップ {i+1}/5: {query}の分析中..."
    
    # 最終結果
    yield f"最終結果: {query}の分析が完了しました！"

with gr.Blocks() as demo:
    query_input = gr.Textbox(label="分析対象")
    result_output = gr.Textbox(label="分析結果")
    
    analyze_btn = gr.Button("分析開始")
    
    analyze_btn.click(
        fn=stream_results,
        inputs=query_input,
        outputs=result_output
    )

demo.launch()
```

### Gradioのキャンセル機能を実装する例

```python
import gradio as gr
import asyncio

async def long_running_task(progress=gr.Progress()):
    try:
        total_steps = 20
        for i in range(total_steps):
            # キャンセル可能なスリープ
            await asyncio.sleep(0.5)
            progress(i/total_steps, desc=f"ステップ {i+1}/{total_steps}")
        return "処理完了"
    except asyncio.CancelledError:
        return "処理がキャンセルされました"

with gr.Blocks() as demo:
    output = gr.Textbox(label="結果")
    
    with gr.Row():
        start_btn = gr.Button("処理開始")
        cancel_btn = gr.Button("キャンセル")
    
    task = start_btn.click(
        fn=long_running_task,
        outputs=output
    )
    
    # キャンセルボタンにタスクをキャンセルする機能を追加
    cancel_btn.click(
        fn=None,
        inputs=None,
        outputs=None, 
        cancels=[task]  # このタスクをキャンセル
    )

demo.launch()
```

## よくある問題と対処法

### 1. モデルのメモリ不足エラー

**問題**: `CUDA out of memory` エラーが発生する

**解決策**:
- モデル精度を下げる（例：fp16）
- バッチサイズを小さくする
- より小さな解像度で生成する
- 余分なモデルをメモリから解放する

### 2. 非同期処理のデッドロック

**問題**: 非同期処理が完了しない、または応答がない

**解決策**:
- `asyncio.sleep(0)` を定期的に挿入してイベントループを解放する
- 重い処理は `asyncio.to_thread()` を使用して別スレッドで実行する
- 非同期関数内で同期ブロッキング呼び出しを避ける

### 3. Gradioのキャッシュ問題

**問題**: UI更新が反映されない

**解決策**:
- ブラウザのキャッシュをクリアする
- `gr.update()` を使用して明示的に更新をトリガーする
- デバッグモードで起動する (`--debug` フラグ)

## テスト実行

```bash
# 全テストを実行
poetry run pytest

# 特定のテストを実行
poetry run pytest tests/generator/test_async_wrapper.py
```

## リンター実行

```bash
# mypyによる型チェック
poetry run mypy src

# flake8による構文チェック
poetry run flake8 src
```

## 参考資料

- [Gradio 公式ドキュメント](https://www.gradio.app/docs)
- [Asyncio 公式ドキュメント](https://docs.python.org/3/library/asyncio.html)
- [Pydantic v2 公式ドキュメント](https://docs.pydantic.dev/latest/)