"""
JSONベースのプロンプト構築機能

このモジュールでは、JSONデータ構造を利用して、
複数の要素を組み合わせた画像生成プロンプトを構築します。
"""

import os
import json
import random
import logging
from pathlib import Path
from typing import Dict, List, Optional, Union

# ロギング設定
logger = logging.getLogger(__name__)

# プロジェクトのルートディレクトリ
project_root = Path(__file__).parent.parent.parent.absolute()

class JSONPromptBuilder:
    """JSONデータからプロンプトを構築するクラス"""
    
    def __init__(self, template_path: Optional[str] = None):
        """初期化メソッド
        
        Args:
            template_path (str, optional): プロンプト要素の定義JSONファイルのパス。
                指定がない場合はデフォルトパスを使用。
        """
        self.template_path = template_path or os.path.join(project_root, "src", "prompt", "elements.json")
        self.template = {}
        
        # テンプレートの読み込み
        self._load_template()
    
    def _load_template(self):
        """プロンプトテンプレートの定義を読み込む"""
        try:
            if os.path.exists(self.template_path):
                with open(self.template_path, "r", encoding="utf-8") as f:
                    self.template = json.load(f)
                    logger.info(f"プロンプトテンプレートを読み込みました: {len(self.template)} カテゴリ")
            else:
                logger.warning(f"プロンプトテンプレートファイルが見つかりません: {self.template_path}")
                # デフォルト値の設定
                self.template = {
                    "subjects": ["portrait", "landscape", "still life", "abstract"],
                    "styles": ["realistic", "fantasy", "anime", "oil painting", "watercolor"],
                    "qualities": ["detailed", "high quality", "masterpiece", "best quality"],
                    "lighting": ["natural lighting", "golden hour", "studio lighting", "dramatic"],
                    "cameras": ["Canon EOS R5", "Nikon Z9", "Sony A7R IV"],
                    "lenses": ["85mm f/1.2", "50mm f/1.4", "35mm f/1.8"]
                }
        except Exception as e:
            logger.error(f"プロンプトテンプレート読み込みエラー: {e}")
            self.template = {}
    
    def build_prompt(self, subject: str = "", style: str = "", quality: str = "",
                    lighting: str = "", camera: str = "", lens: str = "") -> str:
        """基本的なプロンプトを構築する
        
        Args:
            subject (str): 被写体
            style (str): スタイル
            quality (str): 品質
            lighting (str): ライティング
            camera (str): カメラ
            lens (str): レンズ
        
        Returns:
            str: 構築されたプロンプト
        """
        prompt_parts = []
        
        if subject and style:
            prompt_parts.append(f"{style} {subject}")
        elif subject:
            prompt_parts.append(subject)
        elif style:
            prompt_parts.append(style)
        
        if quality:
            prompt_parts.append(quality)
        if lighting:
            prompt_parts.append(lighting)
        if camera:
            prompt_parts.append(camera)
        if lens:
            prompt_parts.append(lens)
        
        return ", ".join(prompt_parts)
    
    def random_prompt(self, include_categories: Optional[List[str]] = None) -> str:
        """ランダムなプロンプトを生成する
        
        Args:
            include_categories (List[str], optional): 含めるカテゴリのリスト
        
        Returns:
            str: 生成されたプロンプト
        """
        if not include_categories:
            include_categories = list(self.template.keys())
        
        prompt_parts = []
        for category in include_categories:
            if category in self.template and self.template[category]:
                prompt_parts.append(random.choice(self.template[category]))
        
        return ", ".join(prompt_parts)
    
    def build_prompt_with_weights(self, elements: Dict[str, str],
                                weights: Dict[str, float]) -> str:
        """重み付きプロンプトを構築する
        
        Args:
            elements (Dict[str, str]): プロンプト要素の辞書
            weights (Dict[str, float]): 要素の重み付け辞書
        
        Returns:
            str: 重み付きプロンプト
        """
        prompt_parts = []
        
        for key, value in elements.items():
            if value:
                if key in weights:
                    prompt_parts.append(f"({value}:{weights[key]})")
                else:
                    prompt_parts.append(value)
        
        return ", ".join(prompt_parts)
    
    def save_template(self, path: str) -> bool:
        """テンプレートを保存する
        
        Args:
            path (str): 保存先のパス
        
        Returns:
            bool: 保存成功時はTrue、失敗時はFalse
        """
        try:
            with open(path, "w") as f:
                json.dump(self.template, f, indent=2)
            logger.info(f"テンプレートを保存しました: {path}")
            return True
        except Exception as e:
            logger.error(f"テンプレート保存エラー: {e}")
            return False 