# アプリケーションフロー

このドキュメントでは、GenerativeAIArtWebアプリケーションの主要なユーザーフローと画面遷移について詳しく説明します。

## ユーザージャーニー概要

GenerativeAIArtWebを利用するユーザーの一般的なジャーニーマップは以下の通りです：

```mermaid
journey
    title GenerativeAIArtWebユーザージャーニー
    section 初期ステップ
      アプリケーション起動: 5: ユーザー
      プロンプト入力: 3: ユーザー
      パラメータ調整: 3: ユーザー
    section 生成プロセス
      画像生成: 4: ユーザー, システム
      進捗確認: 3: ユーザー, システム
      結果確認: 5: ユーザー
    section 後処理
      画像加工: 4: ユーザー
      画像保存: 5: ユーザー
      設定保存: 4: ユーザー
```

## 主要なユーザーフロー

### 1. 基本的な画像生成フロー

最も一般的なユーザーフローは、テキストプロンプトから画像を生成する基本的なプロセスです：

```mermaid
graph TD
    A[アプリケーション起動] --> B[生成画面表示]
    B --> C[プロンプト入力]
    C --> D[パラメータ調整]
    D --> E[生成ボタンクリック]
    E --> F{モデル読込済み?}
    F -->|はい| H[画像生成処理]
    F -->|いいえ| G[モデル読み込み]
    G --> H
    H --> I[進捗表示]
    I --> J[生成完了]
    J --> K[結果表示]
    K --> L{次のアクション?}
    L -->|保存| M[画像保存]
    L -->|加工| N[画像加工]
    L -->|新規生成| C
    
    style E fill:#bbf,stroke:#333,stroke-width:2px
    style H fill:#fbf,stroke:#333,stroke-width:2px
    style J fill:#bfb,stroke:#333,stroke-width:2px
```

### 2. プロンプト生成支援フロー

ユーザーがLLMやJSONベースのプロンプト生成ツールを使用するフロー：

```mermaid
graph TD
    A[プロンプト生成タブ選択] --> B{生成方法?}
    
    B -->|LLM生成| C[基本プロンプト入力]
    C --> D[LLMモデル選択]
    D --> E[スタイル選択]
    E --> F[LLM生成ボタンクリック]
    F --> G[LLMプロンプト生成処理]
    G --> H[生成プロンプト表示]
    
    B -->|JSONビルダー| I[被写体選択]
    I --> J[スタイル選択]
    J --> K[追加要素選択]
    K --> L[ライティング選択]
    L --> M[JSONプロンプト構築]
    M --> N[構築プロンプト表示]
    
    H --> O[プロンプト編集]
    N --> O
    O --> P[プロンプト確定]
    P --> Q[生成タブに適用]
    
    style F fill:#bbf,stroke:#333,stroke-width:2px
    style M fill:#bbf,stroke:#333,stroke-width:2px
    style P fill:#bfb,stroke:#333,stroke-width:2px
```

### 3. 画像後処理フロー

生成された画像に後処理を適用するフロー：

```mermaid
graph TD
    A[生成画像表示] --> B{後処理選択}
    
    B -->|アップスケール| C[解像度設定]
    C --> D[アップスケール実行]
    D --> E[高解像度画像表示]
    
    B -->|ウォーターマーク| F[ウォーターマークテキスト入力]
    F --> G[位置・不透明度設定]
    G --> H[ウォーターマーク追加]
    H --> I[ウォーターマーク付き画像表示]
    
    E --> J{さらに処理?}
    I --> J
    J -->|はい| B
    J -->|いいえ| K[最終画像]
    
    K --> L[カスタム保存]
    L --> M[ファイル名・保存先設定]
    M --> N[メタデータ付き保存]
    N --> O[保存完了通知]
    
    style D fill:#bbf,stroke:#333,stroke-width:2px
    style H fill:#bbf,stroke:#333,stroke-width:2px
    style N fill:#bbf,stroke:#333,stroke-width:2px
```

### 4. 設定管理フロー

ユーザー設定の保存と読み込みのフロー：

```mermaid
graph TD
    A[設定状態] --> B{アクション?}
    
    B -->|保存| C[プリセット名入力]
    C --> D[設定保存]
    D --> E[保存完了通知]
    
    B -->|読み込み| F[プリセット選択]
    F --> G[設定読み込み]
    G --> H[パラメータ更新]
    
    B -->|リセット| I[デフォルト設定復元]
    I --> H
    
    H --> A
    E --> A
    
    style D fill:#bbf,stroke:#333,stroke-width:2px
    style G fill:#bbf,stroke:#333,stroke-width:2px
    style I fill:#bbf,stroke:#333,stroke-width:2px
```

## 画面遷移

アプリケーションの主要な画面と遷移を示します：

```mermaid
stateDiagram-v2
    [*] --> MainScreen
    
    MainScreen --> ImageGenerationProcess: 生成ボタンクリック
    ImageGenerationProcess --> ResultScreen: 生成完了
    ResultScreen --> MainScreen: 新規生成
    
    MainScreen --> PromptScreen: プロンプト生成タブ
    PromptScreen --> MainScreen: プロンプト適用
    
    ResultScreen --> PostProcessingScreen: 後処理オプション
    PostProcessingScreen --> ResultScreen: 処理適用
    
    ResultScreen --> SaveOptionsScreen: 保存オプション
    SaveOptionsScreen --> ResultScreen: 保存完了
    
    MainScreen --> SettingsScreen: 設定タブ
    SettingsScreen --> MainScreen: 設定適用
    
    state MainScreen {
        [*] --> GenerationTab
        GenerationTab --> PromptTab
        PromptTab --> SettingsTab
        SettingsTab --> GenerationTab
    }
    
    state PromptScreen {
        [*] --> LLMPromptTab
        LLMPromptTab --> JSONPromptTab
        JSONPromptTab --> LLMPromptTab
    }
    
    state PostProcessingScreen {
        [*] --> UpscaleOptions
        UpscaleOptions --> WatermarkOptions
        WatermarkOptions --> UpscaleOptions
    }
```

## 主要画面の目的

各画面の主な目的と提供する機能を説明します：

### メイン画像生成画面

**目的**: プロンプト入力と画像生成パラメータの設定、生成開始

**主な機能**:
- プロンプト入力フィールド
- モデル選択ドロップダウン
- 生成パラメータ設定（ステップ数、CFGスケール、サンプラー、サイズ、シード）
- 生成ボタン
- 設定保存・読み込みオプション

### プロンプト生成画面

**目的**: 効果的なプロンプトの生成支援

**主な機能**:
- LLMベースのプロンプト生成
  - 基本プロンプト入力
  - LLMモデル選択
  - スタイル選択
  - 生成ボタン
- JSONベースのプロンプト構築
  - 被写体選択
  - スタイル選択
  - 要素選択
  - 構築ボタン
- 生成プロンプトの編集
- 画像生成タブへの適用ボタン

### 画像後処理画面

**目的**: 生成された画像の加工と保存

**主な機能**:
- アップスケールオプション
  - 解像度設定（1.5x、2x、4x）
  - アップスケールボタン
- ウォーターマークオプション
  - テキスト入力
  - 位置選択
  - 透明度設定
  - 適用ボタン
- カスタム保存オプション
  - 保存ディレクトリ選択
  - ファイル名パターン設定
  - 保存ボタン

### 設定画面

**目的**: アプリケーション設定の管理

**主な機能**:
- プリセット管理
  - 保存
  - 読み込み
  - 削除
- デフォルト値設定
- エクスポート/インポートオプション

## 分岐フロー

ユーザーの意思決定ポイントでの分岐を示します：

### プロンプト入力方法の選択

```mermaid
flowchart TD
    A[プロンプト入力開始] --> B{入力方法?}
    B -->|テキスト直接入力| C[プロンプト記述]
    B -->|LLM支援| D[LLMタブに移動]
    B -->|JSONビルダー| E[JSONタブに移動]
    B -->|履歴から選択| F[生成履歴表示]
    
    C --> G[プロンプト完成]
    D --> G
    E --> G
    F --> G
    
    G --> H[画像生成へ進む]
```

### エラー処理フロー

生成プロセス中に発生する可能性のあるエラーへの対応：

```mermaid
flowchart TD
    A[生成ボタンクリック] --> B{モデル状態?}
    B -->|読み込み済み| C[生成処理開始]
    B -->|未読み込み| D[モデル読み込み試行]
    
    D --> E{読み込み結果?}
    E -->|成功| C
    E -->|失敗| F[エラーメッセージ表示]
    
    C --> G{生成処理?}
    G -->|成功| H[画像表示]
    G -->|エラー発生| I[エラー詳細分析]
    
    I --> J{エラー種類?}
    J -->|メモリ不足| K[メモリエラーメッセージ]
    J -->|プロンプトエラー| L[プロンプト修正提案]
    J -->|モデルエラー| M[モデル再読み込み提案]
    J -->|その他| N[一般エラーメッセージ]
    
    K --> O[解決策提案]
    L --> O
    M --> O
    N --> O
    
    O --> P[再試行オプション]
```

## インタラクションパターン

ユーザーとアプリケーションの主要なインタラクションパターン：

### 画像生成の進捗表示

```mermaid
sequenceDiagram
    actor User as ユーザー
    participant UI as Gradio UI
    participant Generator as 画像生成器
    
    User->>UI: 生成ボタンクリック
    UI->>Generator: 生成処理開始
    UI->>User: プログレスバー表示
    
    loop 生成ステップ
        Generator->>UI: 進捗状況更新
        UI->>User: プログレスバー更新
    end
    
    Generator->>UI: 最終画像
    UI->>User: 画像表示
```

### 設定の保存と読み込み

```mermaid
sequenceDiagram
    actor User as ユーザー
    participant UI as Gradio UI
    participant Settings as 設定マネージャー
    participant Storage as ストレージ
    
    User->>UI: パラメータ設定
    User->>UI: 「設定保存」ボタンクリック
    UI->>User: プリセット名入力プロンプト
    User->>UI: プリセット名入力
    UI->>Settings: 設定保存リクエスト
    Settings->>Storage: 設定ファイル書き込み
    Storage->>Settings: 保存確認
    Settings->>UI: 保存成功通知
    UI->>User: 保存成功メッセージ
    
    User->>UI: 「設定読み込み」ボタンクリック
    UI->>User: プリセット選択表示
    User->>UI: プリセット選択
    UI->>Settings: 設定読み込みリクエスト
    Settings->>Storage: 設定ファイル読み込み
    Storage->>Settings: 設定データ
    Settings->>UI: 設定データ適用
    UI->>User: UI要素の更新
```

## エラーケースと代替フロー

### 代替フロー: モデル読み込み失敗

```mermaid
graph TD
    A[モデル読み込み開始] --> B{ファイル存在?}
    B -->|はい| C[モデル読み込み試行]
    B -->|いいえ| D[ファイル不在エラー]
    
    C --> E{メモリ十分?}
    E -->|はい| F[モデル初期化]
    E -->|いいえ| G[メモリ不足エラー]
    
    F --> H{初期化成功?}
    H -->|はい| I[モデル読み込み完了]
    H -->|いいえ| J[モデル初期化エラー]
    
    D --> K[エラー処理]
    G --> K
    J --> K
    
    K --> L{対処方法?}
    L -->|モデルパス変更| M[別モデルを選択]
    L -->|メモリ解放| N[他のアプリを閉じる]
    L -->|軽量モード| O[小さいモデルに切替]
    
    M --> P[モデル読み込み再試行]
    N --> P
    O --> P
    P --> B
    
### 代替フロー: プロンプト生成の失敗

```mermaid
graph TD
    A[LLMプロンプト生成開始] --> B{LLMの状態?}
    B -->|利用可能| C[プロンプト生成処理]
    B -->|利用不可| D[LLM利用不可エラー]
    
    C --> E{生成結果?}
    E -->|成功| F[プロンプト表示]
    E -->|失敗| G[プロンプト生成エラー]
    
    D --> H{代替手段?}
    G --> H
    
    H -->|JSONビルダー使用| I[JSONビルダーに切替]
    H -->|手動入力| J[直接プロンプト編集]
    H -->|サンプル使用| K[サンプルプロンプト選択]
    
    I --> L[代替プロンプト生成]
    J --> L
    K --> L
    
    L --> F
```

### 代替フロー: 画像生成キャンセル

```mermaid
graph TD
    A[画像生成開始] --> B[生成処理実行中]
    B --> C{ユーザーアクション}
    
    C -->|キャンセルボタン| D[キャンセルリクエスト]
    C -->|処理完了まで待機| E[生成完了]
    
    D --> F[進行中の処理停止]
    F --> G[リソース解放]
    G --> H[キャンセル完了通知]
    
    H --> I{次のアクション?}
    E --> J{結果の確認}
    
    I -->|パラメータ調整| K[パラメータ変更]
    I -->|プロンプト変更| L[プロンプト編集]
    I -->|中止| M[メイン画面に戻る]
    
    K --> N[再生成]
    L --> N
    N --> A
```

## モバイル利用フロー

レスポンシブデザインによるモバイルでの利用フロー：

```mermaid
graph TD
    A[モバイルアクセス] --> B[レスポンシブUI表示]
    B --> C{画面サイズ最適化}
    
    C -->|小画面| D[シングルカラムレイアウト]
    C -->|中画面| E[2カラムレイアウト]
    C -->|大画面| F[フルレイアウト]
    
    D --> G[タブナビゲーション]
    E --> G
    F --> G
    
    G --> H{タブ選択}
    H -->|生成| I[画像生成UI]
    H -->|プロンプト| J[プロンプト生成UI]
    H -->|設定| K[設定UI]
    
    I --> L[タッチ操作による生成]
    J --> M[タッチ操作によるプロンプト編集]
    K --> N[タッチ操作による設定変更]
```

## データフロー概要

主要なデータがアプリケーション内でどのように流れるかを示します：

```mermaid
graph LR
    User((ユーザー)) -->|プロンプト入力| PromptProcessor[プロンプト処理]
    PromptProcessor -->|最適化プロンプト| Generator[画像生成エンジン]
    User -->|生成パラメータ設定| Generator
    
    Generator -->|生成画像| PostProcessor[後処理]
    PostProcessor -->|処理済み画像| FileManager[ファイル管理]
    
    User -->|保存設定| FileManager
    FileManager -->|保存結果| User
    
    User -->|設定保存| SettingsManager[設定管理]
    SettingsManager -->|設定読み込み| User
    
    subgraph データストア
        ImageFiles[(画像ファイル)]
        MetadataFiles[(メタデータ)]
        SettingsFiles[(設定ファイル)]
    end
    
    FileManager -->|画像保存| ImageFiles
    FileManager -->|メタデータ保存| MetadataFiles
    SettingsManager -->|設定保存| SettingsFiles
    SettingsFiles -->|設定読み込み| SettingsManager
```

## まとめ

GenerativeAIArtWebアプリケーションのユーザーフローは、直感的な操作と効率的なワークフローを提供することを目的としています。主要な機能はタブによって整理されており、ユーザーは画像生成、プロンプト生成、後処理、設定管理といった一連のプロセスをスムーズに進めることができます。

エラーケースや代替フローも考慮されており、モデル読み込み失敗、プロンプト生成の問題、処理キャンセルなど、様々な状況に対応できるように設計されています。また、モバイル環境でもレスポンシブなUIによって快適な操作が可能です。

このアプリケーションフローは、エンドユーザーがAI画像生成の技術的な複雑さを意識することなく、クリエイティブな作業に集中できるよう設計されています。