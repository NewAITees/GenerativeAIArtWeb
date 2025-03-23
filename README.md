# GenerativeAIArtWeb

Stable Diffusion 3.5モデルを利用した画像生成Webアプリケーション。SD3.5の最新モデルを使用して、テキストプロンプトから高品質な画像を生成できます。

## 今後の開発予定
- Gradio
- ollama を使ったLLMの使用
- テキストジェネレーター
- Pillow を使った画像の修正
- 解像度の向上用のモデルの追加

## プロジェクト範囲

### 1. Webインターフェース
- Gradio を使用したWeb UIの構築
- ユーザーフレンドリーな直感的インターフェース
- リアルタイムプレビューと設定の簡単な調整機能

### 2. プロンプトの自動生成機能
#### 2.1 LLMを使った生成機能
- 大規模言語モデルによるプロンプトの自動生成・拡張
- シンプルな入力からの詳細なプロンプト作成

#### 2.2 JSONベースの組み合わせ機能
- 外部JSONからテキスト要素を組み合わせてプロンプト作成
- テンプレートベースのプロンプト構築システム

### 3. 解像度アップスケール機能
- 生成後に画像解像度を向上させる機能
- 複数の解像度オプションと画質設定

### 4. カスタムファイル保存機能
- 生成画像を任意のフォルダパスに保存
- ファイル名のパターン設定と自動整理機能

### 5. ウォーターマーク機能
- 生成画像へのカスタムウォーターマーク追加
- ウォーターマークの位置、サイズ、透明度の調整

### 6. 設定保存/読み出し機能
- ユーザー設定のプロファイルとしての保存
- 設定プリセットの管理と再利用

## プロジェクト構成

```
GenerativeAIArtWeb/
├── src/
│   ├── generator/
│   │   ├── sd3_inf.py        # SD3インフェレンスの主要ロジック
│   │   ├── sd3_impls.py      # SD3コアのdiffusionモデルとVAE実装
│   │   ├── dit_embedder.py   # ControlNetエンベッダー
│   │   ├── mmditx.py         # MMDiTモデルコンポーネント
│   │   └── other_impls.py    # CLIP、T5などの関連モデル
│   ├── web/
│   │   ├── app.py            # Gradioのメインアプリ
│   │   ├── routes.py         # ルーティング定義
│   │   ├── templates/        # HTMLテンプレート
│   │   └── static/           # CSS、JavaScript、その他の静的ファイル
│   ├── prompt/
│   │   ├── llm_generator.py  # LLMベースのプロンプト生成
│   │   └── json_builder.py   # JSONベースのプロンプト構築
│   ├── utils/
│   │   ├── upscaler.py       # 解像度向上ユーティリティ
│   │   ├── file_manager.py   # ファイル保存管理
│   │   ├── watermark.py      # ウォーターマーク処理
│   │   └── settings.py       # 設定管理ユーティリティ
├── test/
│   ├── generator/
│   │   ├── test_sd3_inf.py           # SD3インフェレンスのテスト
│   │   ├── test_sd3_impls.py         # SD3コア実装のテスト
│   │   ├── test_dit_embedder.py      # ControlNetエンベッダーのテスト
│   │   ├── test_mmditx.py            # MMDiTコンポーネントのテスト
│   │   ├── test_other_impls.py       # 関連モデルのテスト
│   │   └── test_sd3_integration.py   # 結合テスト
│   ├── web/
│   │   ├── test_app.py               # Webアプリのテスト
│   │   └── test_routes.py            # ルートのテスト
│   ├── prompt/
│   │   ├── test_llm_generator.py     # LLMプロンプト生成のテスト
│   │   └── test_json_builder.py      # JSONプロンプト構築のテスト
│   └── utils/
│       ├── test_upscaler.py          # アップスケーラーのテスト
│       ├── test_file_manager.py      # ファイル管理のテスト
│       ├── test_watermark.py         # ウォーターマーク機能のテスト
│       └── test_settings.py          # 設定管理のテスト
└── pyproject.toml                    # プロジェクト設定
```

## 必要条件

- Python 3.11以上
- Poetry (依存関係管理)
- CUDA対応のGPU（モデルの実行に推奨）

## 依存パッケージ

本プロジェクトでは以下の主要な依存パッケージを使用しています：

- transformers (>=4.49.0,<5.0.0)
- torch (>=2.6.0,<3.0.0)
- torchvision (>=0.21.0,<0.22.0)
- numpy (>=2.2.3,<3.0.0)
- fire (>=0.7.0,<0.8.0)
- pillow (>=8.0.0,<11.0.0)
- einops (>=0.8.1,<0.9.0)
- sentencepiece (>=0.2.0,<0.3.0)
- protobuf (>=5.29.3,<6.0.0)
- webdataset (>=0.2.111,<0.3.0)
- gradio (>=4.31.0,<5.0.0)
- flask (>=3.0.0,<4.0.0)
- requests (>=2.31.0,<3.0.0)
- pyyaml (>=6.0.1,<7.0.0)
- opencv-python (>=4.10.0,<5.0.0)
- python-dotenv (>=1.0.1,<2.0.0)

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
- `llm_model.safetensors` (プロンプト生成用LLM)

## 使用方法

### Webインターフェースの起動

```bash
poetry run python src/web/app.py
```

Gradioを使用する場合（デフォルト）：

```bash
poetry run python src/web/app.py --ui gradio
```

### 画像生成の実行 (CLI)

```bash
poetry run python src/generator/sd3_inf.py \
  --prompt "a photo of a cat" \
  --model "models/sd3.5_large.safetensors" \
  --out_dir "outputs" \
  --steps 40 \
  --cfg_scale 4.5
```

### パラメータ

- `prompt`: 生成する画像の説明テキスト
- `model`: 使用するモデルファイルのパス
- `out_dir`: 出力先ディレクトリ
- `steps`: 生成ステップ数（多いほど高品質だが時間がかかる）
- `cfg_scale`: CFGスケール値（4.0〜5.0が推奨）
- `sampler`: サンプラー（"euler"または"dpmpp_2m"）
- `width`: 画像の幅（デフォルト: 1024）
- `height`: 画像の高さ（デフォルト: 1024）
- `seed`: 乱数シード（再現性のため）
- `upscale`: 解像度向上の倍率（1.0、1.5、2.0、4.0）
- `watermark`: ウォーターマークの有効/無効
- `watermark_text`: ウォーターマークのテキスト
- `watermark_opacity`: ウォーターマークの不透明度（0.0〜1.0）
- `save_dir`: カスタム保存ディレクトリ
- `save_filename`: カスタムファイル名パターン

## 機能詳細

### 1. Webインターフェース
- 画像生成パラメータの直感的な調整
- リアルタイムプレビュー
- 生成履歴の表示と再利用
- モバイル対応レスポンシブデザイン

### 2. プロンプト自動生成
- アイデアからの詳細なプロンプト生成
- スタイル、ムード、写真技術などの要素を自動的に追加
- JSONファイルからのテンプレートベースプロンプト構築
- プロンプト要素の組み合わせと重み付け

### 3. 解像度アップスケール
- 生成後の画像解像度向上（最大4倍）
- 詳細保持とノイズ除去の調整
- バッチ処理対応

### 4. カスタムファイル保存
- 日付、タグ、プロンプトに基づくフォルダ構造
- 自動ファイル名生成
- メタデータの保存

### 5. ウォーターマーク
- テキスト、画像、ロゴのウォーターマーク
- 位置、サイズ、透明度の調整
- バッチ処理でのウォーターマーク適用

### 6. 設定管理
- ユーザー設定のプロファイル保存
- プリセットの作成と管理
- 設定のエクスポート/インポート

## テスト

### テストの実行

すべてのテストを実行：

```bash
poetry run python -m pytest -xvs
```

特定のテストファイルを実行：

```bash
poetry run python -m pytest -xvs test/generator/test_sd3_inf.py
```

特定のテストクラスを実行：

```bash
poetry run python -m pytest -xvs test/generator/test_sd3_inf.py::TestSD3InferencerMock
```

特定のテストメソッドを実行：

```bash
poetry run python -m pytest -xvs test/generator/test_sd3_inf.py::TestSD3InferencerMock::test_initialization
```

### テスト構成

本プロジェクトでは、以下のテスト構成を採用しています：

1. **単体テスト**: 各モジュールの機能を個別にテスト
   - コンポーネントごとの機能検証
   - モックを使用した依存関係の分離

2. **結合テスト**: 異なるコンポーネント間の連携をテスト
   - 画像生成パイプライン、サンプラー、画像処理の結合テスト
   - Webインターフェースとバックエンドの連携

3. **モックを使用したテスト戦略**:
   - モデルファイルがなくてもテストが実行できるよう、外部依存をモック化
   - 統合テストでは実際のモデルファイルが必要な場合をスキップするように設計
   - CI/CD環境で実行可能なテスト環境の提供

## ライセンス

このプロジェクトは MIT の下で公開されています。

## 謝辞

- Stability AI - SD3.5モデルの開発
- Google - T5モデルの開発
- OpenAI - CLIPモデルの開発

## データバリデーションと型チェック

このプロジェクトはデータバリデーションと型チェックに以下を使用しています：

- **pydantic v2**: リクエスト、設定、モデルデータの検証
- **mypy**: 静的型チェック

### mypy型チェックの実行

```bash
poetry run mypy src
```

### pydanticモデルの使用

主なpydanticモデルは`src/models/`ディレクトリで定義されています：

- `base.py`: 基本モデル定義
- `settings.py`: アプリケーション設定モデル
- `file_manager.py`: ファイル管理モデル
- `web.py`: Webアプリケーションモデル

### 型チェックとバリデーションの利点

1. **早期のバグ発見**
   - 実行前に型の問題を発見
   - データバリデーションによる不正な値の防止

2. **自己文書化されたコード**
   - 型ヒントによる明確なインターフェース
   - pydanticモデルの`Field`説明による詳細なドキュメント

3. **IDE補完の強化**
   - より正確なコード補完
   - 型関連のエラーをリアルタイムで表示

4. **堅牢性の向上**
   - 実行時の型エラーを防止
   - データの整合性を保証

### 使用例

設定の読み込みと検証：

```python
from src.models.settings import AppSettings

# 設定をロード
settings_dict = load_settings_from_file()
app_settings = AppSettings.model_validate(settings_dict)

# 型安全な設定へのアクセス
steps = app_settings.generation.steps  # int型が保証される
cfg_scale = app_settings.generation.cfg_scale  # float型が保証される
```

ファイル名の生成：

```python
from src.models.file_manager import FilenameConfig, FileExtension

# 設定を作成
config = FilenameConfig(
    prefix="generated",
    include_date=True,
    extension=FileExtension.PNG
)

# ファイル名を生成
filename = file_manager.generate_filename("my prompt", config)
```

### 開発者向けガイド

1. 新しいモデルの追加
   - `src/models/`に新しいモデルファイルを作成
   - 適切な型ヒントとバリデーションを定義
   - テストを追加

2. 既存コードの型チェック
   - `mypy.ini`で段階的な型チェックを設定
   - `poetry run mypy src`で型チェックを実行
   - エラーを修正

3. バリデーションの追加
   - pydanticの`Field`でバリデーションルールを定義
   - カスタムバリデータを使用して複雑なルールを実装
   - エラーメッセージをカスタマイズ