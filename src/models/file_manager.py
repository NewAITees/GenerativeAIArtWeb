"""
ファイル管理モデル定義

このモジュールでは、ファイル管理に関連するpydanticモデルを定義します。
"""
from datetime import datetime
from enum import Enum
from typing import Dict, List, Optional, Union, Literal
from pydantic import BaseModel, Field, field_validator
from .base import BaseAppModel


class FileExtension(str, Enum):
    """サポートされるファイル拡張子"""
    PNG = "png"
    JPG = "jpg"
    JPEG = "jpeg"
    GIF = "gif"
    BMP = "bmp"


class FileMetadata(BaseAppModel):
    """ファイルメタデータモデル"""
    
    prompt: str = Field(
        default="",
        description="生成プロンプト"
    )
    steps: int = Field(
        default=40,
        description="生成ステップ数"
    )
    cfg_scale: float = Field(
        default=4.5,
        description="CFGスケール"
    )
    sampler: str = Field(
        default="euler",
        description="使用したサンプラー"
    )
    seed: Optional[int] = Field(
        default=None,
        description="生成シード"
    )
    width: int = Field(
        default=1024,
        description="画像の幅"
    )
    height: int = Field(
        default=1024,
        description="画像の高さ"
    )
    created_at: datetime = Field(
        default_factory=datetime.now,
        description="作成日時"
    )
    custom_tags: List[str] = Field(
        default_factory=list,
        description="カスタムタグ"
    )


class FilenameConfig(BaseAppModel):
    """ファイル名生成設定モデル"""
    
    prefix: str = Field(
        default="",
        description="ファイル名の接頭辞"
    )
    include_date: bool = Field(
        default=True,
        description="日付を含めるかどうか"
    )
    include_time: bool = Field(
        default=True,
        description="時刻を含めるかどうか"
    )
    extension: FileExtension = Field(
        default=FileExtension.PNG,
        description="ファイル拡張子"
    )
    
    @field_validator('extension')
    @classmethod
    def validate_extension(cls, v: Union[str, FileExtension]) -> FileExtension:
        """拡張子のバリデーション"""
        if isinstance(v, str):
            if v.startswith('.'):
                v = v[1:]
            try:
                return FileExtension(v.lower())
            except ValueError:
                return FileExtension.PNG
        return v 