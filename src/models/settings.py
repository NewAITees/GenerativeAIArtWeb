"""
設定モデル定義

このモジュールでは、アプリケーション設定に関連するpydanticモデルを定義します。
"""
from typing import Dict, List, Optional, Union, Literal
from pydantic import BaseModel, Field, field_validator, model_validator
from .base import BaseAppModel


class ImageGenerationSettings(BaseAppModel):
    """画像生成の設定モデル"""
    
    prompt: str = Field(
        default="",
        description="画像生成のプロンプト"
    )
    steps: int = Field(
        default=40, 
        ge=10, 
        le=100,
        description="生成ステップ数（10〜100）"
    )
    cfg_scale: float = Field(
        default=4.5, 
        ge=1.0, 
        le=10.0,
        description="CFGスケール値（1.0〜10.0）"
    )
    sampler: Literal["euler", "dpmpp_2m"] = Field(
        default="euler",
        description="使用するサンプラー"
    )
    width: int = Field(
        default=1024, 
        ge=256, 
        le=1536,
        description="生成画像の幅（256〜1536）"
    )
    height: int = Field(
        default=1024, 
        ge=256, 
        le=1536,
        description="生成画像の高さ（256〜1536）"
    )
    seed: Optional[int] = Field(
        default=None,
        description="生成シード値（指定なしはランダム）"
    )
    
    # バリデーション: 幅と高さは64の倍数に調整
    @field_validator('width', 'height', mode='after')
    @classmethod
    def validate_dimensions(cls, v: int) -> int:
        # 64の倍数に丸める
        if v % 64 != 0:
            v = (v // 64) * 64
        return v


class WatermarkSettings(BaseAppModel):
    """ウォーターマーク設定モデル"""
    
    enabled: bool = Field(
        default=False,
        description="ウォーターマークの有効/無効"
    )
    text: str = Field(
        default="Generated with SD3.5",
        description="ウォーターマークのテキスト"
    )
    position: Literal["top-left", "top-right", "bottom-left", "bottom-right", "center"] = Field(
        default="bottom-right",
        description="ウォーターマークの位置"
    )
    opacity: float = Field(
        default=0.3,
        ge=0.0,
        le=1.0,
        description="ウォーターマークの不透明度"
    )


class FileSettings(BaseAppModel):
    """ファイル管理設定モデル"""
    
    save_directory: str = Field(
        default="outputs",
        description="保存ディレクトリ"
    )
    filename_pattern: str = Field(
        default="{prompt}_{date}",
        description="ファイル名パターン"
    )


class AppSettings(BaseAppModel):
    """アプリケーション全体の設定モデル"""
    
    generation: ImageGenerationSettings = Field(
        default_factory=ImageGenerationSettings,
        description="画像生成設定"
    )
    watermark: WatermarkSettings = Field(
        default_factory=WatermarkSettings,
        description="ウォーターマーク設定"
    )
    file: FileSettings = Field(
        default_factory=FileSettings,
        description="ファイル設定"
    )
    
    # 追加のフィールド
    upscale: float = Field(
        default=1.0,
        description="アップスケール倍率"
    )
    
    # モデル全体の検証
    @model_validator(mode='after')
    def validate_settings(self) -> 'AppSettings':
        # 特定の条件をチェックする例
        if self.generation.steps > 50 and self.generation.width > 1024:
            # 警告を出すことも可能（例外は投げない）
            self.generation.steps = 50
        return self 