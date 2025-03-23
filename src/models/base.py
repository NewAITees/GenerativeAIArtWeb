"""
ベースモデル定義

このモジュールでは、プロジェクト全体で使用される共通のpydanticモデルを定義します。
"""
from typing import Any, Dict, List, Optional, Union
from pydantic import BaseModel, ConfigDict, Field


class BaseAppModel(BaseModel):
    """アプリケーション全体で使用される基本モデル"""
    
    # pydantic v2の設定スタイル
    model_config = ConfigDict(
        validate_assignment=True,  # 属性代入時も検証
        extra="forbid",            # 未定義フィールドは禁止
        str_strip_whitespace=True, # 文字列のホワイトスペース除去
        validate_default=True,     # デフォルト値も検証
        protected_namespaces=(),   # プロテクト名前空間なし
    ) 