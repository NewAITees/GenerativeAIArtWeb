# GenerativeAIArtWeb

Stable Diffusion 3.5モデルを利用した画像生成Webアプリケーション。SD3.5の最新モデルを使用して、テキストプロンプトから高品質な画像を生成できます。

## 主な特徴

- **直感的なWebインターフェース**: Gradioを使用した使いやすいUI
- **高品質な画像生成**: SD3.5モデルによる最先端の画像生成
- **プロンプト生成支援**: LLMとJSONベースの二種類のプロンプト生成ツール
- **後処理機能**: アップスケール、ウォーターマーク、カスタムファイル保存
- **設定管理**: ユーザー設定のプロファイル保存と読み込み

## プロジェクト構成

```
GenerativeAIArtWeb/
├── src/
│   ├── generator/        # 画像生成エンジン
│   ├── web/              # Webインターフェース
│   ├── prompt/           # プロンプト生成ツール
│   ├── utils/            # ユーティリティ機能
│   └── models/           # Pydanticモデル定義
├── models/               # モデルファイル保存場所
├── outputs/              # 生成画像の出力先
└── settings/             # 設定ファイル保存場所
```

## 必要条件

- Python 3.11以上
- Poetry (依存関係管理)
- CUDA対応のGPU（モデルの実行に推奨）

## インストール方法

Poetry による依存パッケージのインストール：

```bash
# リポジトリのクローン
git clone https://github.com/yourusername/GenerativeAIArtWeb.git
cd GenerativeAIArtWeb

# Poetryがインストールされていない場合
curl -sSL https://install.python-poetry.org | python3 -

# 依存パッケージのインストール
poetry install

# オプション: CUDA サポートを有効にする場合
poetry install -E cuda
```

## モデルファイルの準備

以下のモデルファイルを `models` ディレクトリに配置する必要があります：

- `clip_g.safetensors` (OpenCLIP bigG、SDXLと同じ)
- `clip_l.safetensors` (OpenAI CLIP-L、SDXLと同じ)
- `t5xxl.safetensors` または `t5xxl_fp16.safetensors` (Google T5-v1.1-XXL)
- `sd3.5_large.safetensors` または別のSD3系モデルファイル
- オプション: `sd3_vae.safetensors` (VAEが別ファイルの場合)

## 使用方法

### Webインターフェースの起動

```bash
poetry run python src/web/app.py
```

Gradioインターフェースが起動し、ブラウザでアクセスできます（デフォルトは http://localhost:7860）。

### コマンドラインオプション

```bash
# デバッグモードで起動
poetry run python src/web/app.py --debug

# 公開リンクを生成（他のユーザーがアクセス可能）
poetry run python src/web/app.py --share

# デモモードで起動（実際のモデルを使わない）
poetry run python src/web/demo.py
```

### 主要機能の使い方

- **画像生成**: プロンプトを入力し、生成パラメータを調整して「画像生成」ボタンをクリック
- **プロンプト生成**: 「プロンプト生成」タブでLLMまたはJSONベースのプロンプト生成を使用
- **画像後処理**: 生成された画像に対して、アップスケールやウォーターマークを適用
- **設定保存**: 使用頻度の高い設定をプロファイルとして保存

## 技術スタック

- **Python 3.11+**: 主要開発言語
- **PyTorch 2.6.0+**: ディープラーニングフレームワーク
- **Gradio 4.31.0+**: Webインターフェース構築
- **Pydantic 2.5.2+**: データバリデーションと型チェック
- **Ollama 0.1.5+**: ローカルLLM実行環境
- **Poetry**: 依存関係管理

## ライセンス

このプロジェクトは MIT ライセンスで公開されています。

## 謝辞

- Stability AI - SD3.5モデルの開発
- Google - T5モデルの開発
- OpenAI - CLIPモデルの開発