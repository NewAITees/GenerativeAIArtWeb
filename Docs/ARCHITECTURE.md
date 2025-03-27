# アーキテクチャ設計

## 概要

GenerativeAIArtWebは、Stable Diffusion 3.5モデルを使用してテキストプロンプトから高品質な画像を生成するWebアプリケーションです。このドキュメントでは、システム全体のアーキテクチャと主要コンポーネントについて説明し、特に非同期処理機能に焦点を当てています。

## システムアーキテクチャ

```mermaid
graph TD
    User[ユーザー] -->|リクエスト| WebInterface
    WebInterface[Web インターフェース\nGradio] -->|プロンプト| PromptProcessor
    WebInterface -->|画像生成パラメータ| Generator
    PromptProcessor[プロンプト処理\nLLM/JSON] -->|最適化プロンプト| Generator
    Generator[画像生成エンジン\nSD3.5] -->|生成画像| PostProcessor
    PostProcessor[後処理モジュール\nアップスケール/ウォーターマーク] -->|最終画像| WebInterface
    FileManager[ファイル管理] -->|保存された画像| WebInterface
    Settings[設定管理] -->|ユーザー設定| WebInterface
    WebInterface -->|カスタム保存| FileManager
    WebInterface -->|設定の保存/読込| Settings

    classDef core fill:#f9f,stroke:#333,stroke-width:2px;
    classDef interface fill:#bbf,stroke:#333,stroke-width:2px;
    classDef utility fill:#bfb,stroke:#333,stroke-width:2px;
    classDef async fill:#ffc,stroke:#333,stroke-width:2px,stroke-dasharray: 5 5;
    
    class Generator core;
    class WebInterface interface;
    class PromptProcessor,PostProcessor,FileManager,Settings utility;
    class Generator,WebInterface async;
```

## 非同期処理アーキテクチャ

GenerativeAIArtWebでは、非同期処理を活用してユーザーエクスペリエンスを向上させています。特に、画像生成のような時間のかかる処理を非同期で実行することで、UIの応答性を維持し、進捗状況をリアルタイムで表示できます。

```mermaid
sequenceDiagram
    participant User as ユーザー
    participant UI as Gradio UI
    participant Handler as イベントハンドラ
    participant Generator as 画像生成エンジン
    participant AsyncWorker as 非同期ワーカー

    User->>UI: 画像生成リクエスト
    UI->>Handler: イベント発火
    Handler->>AsyncWorker: 非同期タスク開始
    AsyncWorker->>Generator: 画像生成処理
    AsyncWorker-->>UI: 初期状態更新
    Note over AsyncWorker,UI: UIはレスポンシブに保たれる
    
    loop 生成プロセス
        Generator->>AsyncWorker: 進捗状況
        AsyncWorker-->>UI: 進捗表示更新
    end
    
    Generator->>AsyncWorker: 生成完了
    AsyncWorker-->>UI: 最終画像と結果を表示
    UI-->>User: 生成結果を表示
```

## 主要コンポーネント

### 1. Web インターフェース (src/web)

非同期対応したGradioを使用したウェブインターフェースで、ユーザーが簡単に画像生成パラメータを設定し、結果を表示できます。また、進捗状況もリアルタイムで確認可能です。

#### 主な非同期機能:
- 非同期イベントハンドラによる画像生成処理
- 生成プロセスのリアルタイム進捗表示
- 長時間実行タスクのキャンセル機能
- 複数リクエストの並行処理

### 2. 画像生成エンジン (src/generator)

Stable Diffusion 3.5モデルを中心とした画像生成の中核機能を提供します。非同期対応させることで、効率的な実行が可能になります。

#### 主な構成要素:
- `SD3Inferencer`: 画像生成クラス
- `ProgressCallback`: 進捗状況を通知するコールバックインターフェース
- `StreamingOutput`: 段階的な出力結果をストリーミングするためのクラス

### 3. プロンプト処理 (src/prompt)

テキストプロンプトを最適化するための機能を提供します。LLMとJSONベースの両方のプロンプト生成メカニズムをサポートしています。

### 4. 後処理モジュール (src/utils)

生成された画像の後処理を行うユーティリティを提供します。アップスケールやウォーターマーク追加などの処理も非同期で実行可能です。

### 5. 設定管理 (src/utils)

ユーザー設定のプロファイル保存と管理を行います。

## 非同期データフロー

```mermaid
sequenceDiagram
    participant User as ユーザー
    participant UI as Gradio UI
    participant Generator as SD3 Generator
    participant Callback as ProgressCallback
    participant ErrorHandler as エラーハンドラ
    
    User->>UI: プロンプト入力 & 画像生成開始
    
    UI->>Generator: 非同期生成開始(async generate_image)
    
    activate Generator
    Generator-->>UI: タスク開始通知
    
    par 生成プロセス
        loop サンプリングステップ
            Generator->>Callback: 進捗更新(step/total_steps)
            Callback-->>UI: 進捗バー & ステータス更新
            
            alt キャンセル要求
                User->>UI: キャンセルボタンクリック
                UI->>Generator: キャンセルシグナル送信
                Generator-->>UI: 処理中断 & リソース解放
                UI-->>User: キャンセル完了通知
            end
        end
    and エラー監視
        Generator->>ErrorHandler: エラー発生検知
        ErrorHandler-->>UI: エラー情報表示
        UI-->>User: エラー通知
    end
    
    alt 正常完了
        Note over Generator: プレビュー画像の生成
        Generator->>Callback: 中間プレビュー画像
        Callback-->>UI: プレビュー表示更新
        
        Generator->>Generator: 最終画像の生成
        Generator-->>UI: 生成完了 & 最終画像
        UI-->>User: 生成結果表示
    end
    
    deactivate Generator
```

## 実装の詳細

### 非同期処理の実現方法

Gradioは3.50.0以降、非同期関数を直接イベントハンドラとして使用できるようになりました。これを活用して以下の機能を実装しています：

1. **非同期イベントハンドラ**: `async def` で定義された関数をGradioのイベントハンドラとして登録
2. **ストリーミング出力**: ジェネレータ関数を使用して部分的な結果を逐次的に返す
3. **進捗表示**: `gr.Progress()` コンポーネントを使用して進捗バーを表示
4. **並行処理**: `asyncio.gather()` を使用して複数のタスクを並行実行

### キャンセル機能の実装

長時間実行されるタスクをユーザーがキャンセルできるようにするため、asyncioの`CancelledError`例外を利用します。Gradioのキャンセルボタンと組み合わせることで、ユーザーがいつでも処理を中断できます。

## フォルダ構成

```mermaid
graph TD
    Root[GenerativeAIArtWeb/] --> Src[src/]
    Root --> Test[test/]
    Root --> Models[models/]
    Root --> Outputs[outputs/]

    Src --> Generator[generator/]
    Src --> Web[web/]
    Src --> Prompt[prompt/]
    Src --> Utils[utils/]
    Src --> ModelsDef[models/]

    Generator --> SD3Inf[sd3_inf.py]
    Generator --> SD3Impls[sd3_impls.py]
    Generator --> AsyncWrapper[async_wrapper.py]

    Web --> App[app.py]
    Web --> Routes[routes.py]
    Web --> Templates[templates/]
    Web --> Static[static/]

    Prompt --> LLMGen[llm_generator.py]
    Prompt --> JSONBuilder[json_builder.py]

    Utils --> Upscaler[upscaler.py]
    Utils --> FileManager[file_manager.py]
    Utils --> Watermark[watermark.py]
    Utils --> Settings[settings.py]

    ModelsDef --> Base[base.py]
    ModelsDef --> FMmodels[file_manager.py]
    ModelsDef --> SettingsModels[settings.py]
    ModelsDef --> WebModels[web.py]

    classDef coreComponents fill:#f9f,stroke:#333,stroke-width:1px;
    classDef webComponents fill:#bbf,stroke:#333,stroke-width:1px;
    classDef utilityComponents fill:#bfb,stroke:#333,stroke-width:1px;
    classDef asyncComponents fill:#ffc,stroke:#333,stroke-width:1px,stroke-dasharray: 5 5;
    
    class Generator,SD3Inf,SD3Impls coreComponents;
    class Web,App,Routes,Templates,Static webComponents;
    class Utils,Prompt,Upscaler,FileManager,Watermark,Settings,LLMGen,JSONBuilder utilityComponents;
    class AsyncWrapper asyncComponents;
```

## 技術選定

1. **Gradio 4.31.0+**: シンプルなUI構築と非同期処理のサポート
2. **Asyncio**: Pythonの標準非同期プログラミングライブラリ
3. **Pydantic v2**: 型チェックとデータバリデーションのための堅牢なフレームワーク
4. **Stable Diffusion 3.5**: 高品質な画像生成

## 設計方針と考慮事項

1. **非侵襲的な実装**: 既存コードへの影響を最小限に抑える
2. **段階的な移行**: 一度にすべてを非同期化するのではなく、重要な部分から段階的に移行
3. **ユーザーエクスペリエンスの重視**: 進捗表示やキャンセル機能でUXを向上
4. **エラーハンドリング**: 非同期処理特有のエラーに対する適切な処理
5. **拡張性**: 将来的な機能追加に対応できる柔軟な設計

## テストアーキテクチャ

### テストの構成

```mermaid
graph TD
    Root[tests/] --> Conftest[conftest.py]
    Root --> Init[__init__.py]
    Root --> Generator[generator/]
    Root --> Web[web/]
    Root --> Prompt[prompt/]
    Root --> Utils[utils/]
    
    Generator --> GenUnit[unit/]
    Generator --> GenInt[integration/]
    Generator --> GenFixtures[fixtures.py]
    Generator --> GenConftest[conftest.py]
    
    Web --> WebUnit[unit/]
    Web --> WebInt[integration/]
    Web --> WebE2E[e2e/]
    Web --> WebFixtures[fixtures.py]
    
    Prompt --> PromptUnit[unit/]
    Prompt --> PromptInt[integration/]
    Prompt --> PromptFixtures[fixtures.py]
    
    Utils --> UtilsUnit[unit/]
    Utils --> UtilsInt[integration/]
    Utils --> UtilsFixtures[fixtures.py]
    
    classDef testFiles fill:#f9f,stroke:#333,stroke-width:1px;
    classDef testDirs fill:#bbf,stroke:#333,stroke-width:1px;
    classDef fixtures fill:#bfb,stroke:#333,stroke-width:1px;
    
    class Conftest,Init,GenConftest testFiles;
    class Generator,Web,Prompt,Utils,GenUnit,GenInt,WebUnit,WebInt,WebE2E,PromptUnit,PromptInt,UtilsUnit,UtilsInt testDirs;
    class GenFixtures,WebFixtures,PromptFixtures,UtilsFixtures fixtures;
```

### テストレベル

1. **ユニットテスト (`unit/`)**
   - 個々のコンポーネントの独立したテスト
   - モックを使用して外部依存を分離
   - 高速な実行と即座のフィードバック

2. **統合テスト (`integration/`)**
   - 複数のコンポーネントの連携テスト
   - 実際のコンポーネント間の相互作用を検証
   - 部分的なモックの使用

3. **E2Eテスト (`e2e/`)**
   - エンドツーエンドのユーザーシナリオテスト
   - 実際のUIとバックエンドの統合テスト
   - モックを最小限に抑えた実環境に近いテスト

### テストフィクスチャ

各モジュールの`fixtures.py`には、以下のような共通フィクスチャを定義：

```python
# 例: generator/fixtures.py
@pytest.fixture
def mock_model():
    """モデルのモックフィクスチャ"""
    return Mock(spec=SD3Model)

@pytest.fixture
def sample_prompt():
    """サンプルプロンプトフィクスチャ"""
    return {
        "text": "test prompt",
        "params": {"steps": 20, "cfg_scale": 7.0}
    }
```

### 非同期テスト

非同期コンポーネントのテストには`pytest-asyncio`を使用：

```python
@pytest.mark.asyncio
async def test_async_generation():
    """非同期画像生成のテスト"""
    generator = AsyncImageGenerator()
    result = await generator.generate(prompt="test")
    assert result is not None
```

### テストカバレッジ

テストカバレッジの目標と測定：

```mermaid
pie title テストカバレッジ目標
    "ユニットテスト" : 90
    "統合テスト" : 70
    "E2Eテスト" : 50
```

### CI/CDパイプラインでのテスト実行

```mermaid
graph TD
    A[コードプッシュ] --> B[静的解析]
    B --> C[ユニットテスト]
    C --> D[統合テスト]
    D --> E{カバレッジ確認}
    E -->|基準満たす| F[E2Eテスト]
    E -->|基準未満| G[レビュー要求]
    F -->|成功| H[デプロイ]
    F -->|失敗| G
```

### テスト実行コマンド

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

### モックとスタブ

主要なモックオブジェクト：

```python
# モデルモック
@pytest.fixture
def mock_sd3_model():
    model = Mock()
    model.generate.return_value = create_dummy_image()
    return model

# 非同期モック
@pytest.fixture
async def mock_async_generator():
    generator = AsyncMock()
    generator.generate.return_value = create_dummy_image()
    return generator
```

### テスト環境設定

テスト用の環境変数とコンフィグ：

```python
# conftest.py
@pytest.fixture(autouse=True)
def env_setup():
    """テスト環境のセットアップ"""
    os.environ["TEST_MODE"] = "true"
    os.environ["MODEL_PATH"] = "test_models/"
    yield
    os.environ.pop("TEST_MODE")
    os.environ.pop("MODEL_PATH")
```