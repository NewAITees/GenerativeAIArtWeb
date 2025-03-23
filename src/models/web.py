"""
Webアプリケーションモデル定義

このモジュールでは、Webインターフェースに関連するpydanticモデルを定義します。
"""
from enum import Enum
from typing import Dict, List, Optional, Union, Literal, Any
from pydantic import BaseModel, Field, field_validator, model_validator
from .base import BaseAppModel
from .settings import ImageGenerationSettings, WatermarkSettings


class GenerationRequest(BaseAppModel):
    """画像生成リクエストモデル"""
    
    prompt: str = Field(
        ...,  # 必須項目
        min_length=1,
        max_length=1000,
        description="画像生成のプロンプト"
    )
    model_path: str = Field(
        ...,  # 必須項目
        description="モデルファイルのパス"
    )
    settings: ImageGenerationSettings = Field(
        default_factory=ImageGenerationSettings,
        description="生成設定"
    )
    
    @model_validator(mode='after')
    def validate_request(self) -> 'GenerationRequest':
        # モデルパスの簡易検証
        if not self.model_path.endswith(('.safetensors', '.ckpt', '.pt')):
            raise ValueError("モデルパスの拡張子が無効です")
        return self


class GenerationResponse(BaseAppModel):
    """画像生成レスポンスモデル"""
    
    success: bool = Field(
        default=True,
        description="生成成功フラグ"
    )
    message: str = Field(
        default="画像が生成されました",
        description="結果メッセージ"
    )
    image_path: Optional[str] = Field(
        default=None,
        description="生成画像のパス"
    )
    metadata: Dict[str, Any] = Field(
        default_factory=dict,
        description="生成メタデータ"
    )
    error: Optional[str] = Field(
        default=None,
        description="エラーメッセージ（失敗時）"
    )


class UpscaleRequest(BaseAppModel):
    """アップスケールリクエストモデル"""
    
    image_path: str = Field(
        ...,  # 必須項目
        description="アップスケールする画像のパス"
    )
    scale: float = Field(
        default=2.0,
        ge=1.0,
        le=4.0,
        description="スケール倍率"
    )
    method: Literal["nearest", "box", "bilinear", "bicubic", "lanczos"] = Field(
        default="lanczos",
        description="アップスケール方法"
    )
    output_path: Optional[str] = Field(
        default=None,
        description="出力パス（指定なしの場合は自動生成）"
    )


class UIState(BaseAppModel):
    """UIの状態モデル"""
    
    current_tab: str = Field(
        default="image_generation",
        description="現在のタブ"
    )
    selected_preset: Optional[str] = Field(
        default=None,
        description="選択されたプリセット"
    )
    last_generation_time: Optional[str] = Field(
        default=None,
        description="最後の生成時刻"
    )
    available_models: List[str] = Field(
        default_factory=list,
        description="利用可能なモデル一覧"
    )
    available_presets: List[str] = Field(
        default_factory=list,
        description="利用可能なプリセット一覧"
    ) 