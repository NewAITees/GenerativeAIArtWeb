# GenerativeAIArtWeb

Stable Diffusion 3.5モデルを利用した画像生成Webアプリケーション。SD3.5の最新モデルを使用して、テキストプロンプトから高品質な画像を生成できます。

## 主な特徴

- **直感的なWebインターフェース**: Gradioを使用した使いやすいUI
- **高品質な画像生成**: SD3.5モデルによる最先端の画像生成
- **プロンプト生成支援**: LLMとJSONベースの二種類のプロンプト生成ツール
- **後処理機能**: アップスケール、ウォーターマーク、カスタムファイル保存
- **設定管理**: ユーザー設定のプロファイル保存と読み込み
- **非同期処理**: 画像生成やモデル読み込みの非同期実行とプログレス表示

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

# 仮想環境の作成とPythonバージョンの設定
pyenv install 3.11.7
pyenv local 3.11.7
poetry env use $(pyenv which python)

# 依存パッケージのインストール
poetry install

# オプション: CUDA サポートを有効にする場合
poetry install -E cuda
```

### 依存パッケージのバージョン

主要な依存パッケージとそのバージョン：

- Python: 3.11.7以上
- PyTorch: 2.6.0
- torchvision: 0.21.0
- Gradio: 4.31.0
- Pydantic: 2.5.2
- Ollama: 0.1.5

### 開発環境の設定

1. **開発ツールのインストール**
```bash
# 開発用依存パッケージのインストール
poetry install --with dev

# pre-commit フックのインストール
poetry run pre-commit install
```

2. **環境変数の設定**
```bash
# .env ファイルの作成
cp .env.example .env

# 必要な環境変数を設定
nano .env
```

3. **型チェックの実行**
```bash
# mypyによる型チェック
poetry run mypy src

# vulture による未使用コード検出
poetry run vulture src
```

### トラブルシューティング

1. **CUDA関連の問題**
   - CUDA対応のGPUドライバーがインストールされていることを確認
   - `nvidia-smi` コマンドでGPUの状態を確認
   - VRAM使用量を監視

2. **メモリ不足エラー**
   - バッチサイズを小さくする
   - 画像サイズを調整
   - 不要なプロセスを終了

3. **モデル読み込みエラー**
   - モデルファイルの整合性を確認
   - ディスク容量を確認
   - 必要に応じてモデルを再ダウンロード

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
  - 生成中はプログレスバーで進捗状況を確認可能
  - 長時間の処理中でもUIの操作が可能
- **プロンプト生成**: 「プロンプト生成」タブでLLMまたはJSONベースのプロンプト生成を使用
- **画像後処理**: 生成された画像に対して、アップスケールやウォーターマークを適用
- **設定保存**: 使用頻度の高い設定をプロファイルとして保存

### 非同期処理の仕組み

アプリケーションは以下の処理を非同期で実行します：

- モデルの読み込み
- 画像生成
- 画像のアップスケール
- ウォーターマーク追加
- カスタムファイル保存

各処理は進捗状況をリアルタイムで表示し、UIのレスポンシブ性を維持します。

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

## テスト

### テストの実行

```bash
# 全テストの実行
poetry run pytest

# カバレッジレポート付きで実行
poetry run pytest --cov=src --cov-report=html

# 特定のテストの実行
poetry run pytest tests/generator/unit/

# 非同期テストの実行
poetry run pytest tests/web/integration/

# E2Eテストの実行
poetry run pytest tests/web/e2e/
```

### テストの種類

1. **ユニットテスト** (`tests/*/unit/`)
   - 個々のコンポーネントの独立したテスト
   - 外部依存はモックを使用

2. **統合テスト** (`tests/*/integration/`)
   - 複数のコンポーネントの連携テスト
   - 実際のコンポーネント間の相互作用を検証

3. **E2Eテスト** (`tests/web/e2e/`)
   - エンドツーエンドのユーザーシナリオテスト
   - 実際のUIとバックエンドの統合テスト

### テスト環境の準備

```bash
# テスト用の依存パッケージをインストール
poetry install --with dev

# テスト用の環境変数を設定
cp .env.test.example .env.test
```

詳細なテストのガイドラインとアーキテクチャについては、以下のドキュメントを参照してください：

- [テストアーキテクチャ](docs/ARCHITECTURE.md#テストアーキテクチャ)
- [テストガイドライン](docs/CODE_GUIDELINE.md#テストガイドライン)