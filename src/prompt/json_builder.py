"""
JSONベースのプロンプト構築機能

このモジュールでは、JSONデータ構造を利用して、
複数の要素を組み合わせた画像生成プロンプトを構築する機能を提供します。
"""

import os
import json
import logging
from pathlib import Path

# ロギング設定
logger = logging.getLogger(__name__)

# プロジェクトのルートディレクトリ
project_root = Path(__file__).parent.parent.parent.absolute()


class JSONPromptBuilder:
    """JSONデータからプロンプトを構築するクラス"""
    
    def __init__(self, elements_path=None):
        """初期化メソッド
        
        Args:
            elements_path (str, optional): プロンプト要素の定義JSONファイルのパス。
                指定がない場合はデフォルトパスを使用。
        """
        self.elements_path = elements_path or os.path.join(project_root, "src", "prompt", "elements.json")
        self.elements = {}
        
        # 要素の読み込み
        self._load_elements()
    
    def _load_elements(self):
        """プロンプト要素の定義を読み込む"""
        try:
            if os.path.exists(self.elements_path):
                with open(self.elements_path, "r", encoding="utf-8") as f:
                    data = json.load(f)
                    self.elements = data.get("elements", {})
                    logger.info(f"プロンプト要素を読み込みました: {len(self.elements)} カテゴリ")
            else:
                logger.warning(f"プロンプト要素ファイルが見つかりません: {self.elements_path}")
                # デフォルト値の設定
                self.elements = {
                    "subject": ["portrait", "landscape", "still life", "abstract"],
                    "style": ["realistic", "fantasy", "anime", "oil painting", "watercolor"],
                    "lighting": ["daylight", "sunset", "studio", "dramatic", "neon"],
                    "camera": ["50mm", "portrait lens", "wide angle", "macro", "telephoto"]
                }
        except Exception as e:
            logger.error(f"プロンプト要素読み込みエラー: {e}")
            # エラー時はデフォルト値を使用
            self.elements = {}
    
    def build_prompt(self, element_dict):
        """要素辞書からプロンプトを構築する
        
        Args:
            element_dict (dict): プロンプト構築に使用する要素の辞書
                例: {"subject": "landscape", "style": "fantasy", "elements": ["mountains", "river"]}
        
        Returns:
            str: 構築されたプロンプト
        """
        if not element_dict:
            return ""
        
        try:
            # 基本的な構造を作成
            subject = element_dict.get("subject", "")
            style = element_dict.get("style", "")
            elements_list = element_dict.get("elements", [])
            lighting = element_dict.get("lighting", "")
            
            # プロンプトパーツの構築
            prompt_parts = []
            
            # スタイルと被写体
            if style and subject:
                prompt_parts.append(f"A {style} {subject}")
            elif subject:
                prompt_parts.append(f"A {subject}")
            elif style:
                prompt_parts.append(f"A {style} image")
            
            # 追加要素
            if elements_list:
                elements_str = ", ".join(elements_list)
                if prompt_parts:
                    prompt_parts[0] += f" with {elements_str}"
                else:
                    prompt_parts.append(f"Image with {elements_str}")
            
            # ライティング
            if lighting:
                prompt_parts.append(f"{lighting} lighting")
            
            # クオリティ向上のための追加要素
            # プロンプトの最後に必ず追加
            quality_suffix = "8k resolution, highly detailed, professional photography"
            prompt_parts.append(quality_suffix)
            
            # プロンプトを組み立て
            prompt = ", ".join(prompt_parts)
            
            return prompt
        except Exception as e:
            logger.error(f"プロンプト構築エラー: {e}")
            # エラー時はできるだけ情報を結合して返す
            return ", ".join([str(v) for v in element_dict.values() if v])
    
    def save_elements(self, elements_dict):
        """プロンプト要素の定義を保存する
        
        Args:
            elements_dict (dict): 保存するプロンプト要素の辞書
        
        Returns:
            bool: 保存成功時はTrue、失敗時はFalse
        """
        try:
            # 親ディレクトリの存在確認と作成
            os.makedirs(os.path.dirname(self.elements_path), exist_ok=True)
            
            # 要素の保存
            with open(self.elements_path, "w", encoding="utf-8") as f:
                json.dump({"elements": elements_dict}, f, ensure_ascii=False, indent=2)
            
            # 保存後に要素を更新
            self.elements = elements_dict
            
            logger.info(f"プロンプト要素を保存しました: {self.elements_path}")
            return True
        except Exception as e:
            logger.error(f"プロンプト要素保存エラー: {e}")
            return False
    
    def get_default_elements(self):
        """デフォルトのプロンプト要素を取得する
        
        Returns:
            dict: デフォルトのプロンプト要素辞書
        """
        return {
            "subject": ["portrait", "landscape", "still life", "abstract"],
            "style": ["realistic", "fantasy", "anime", "oil painting", "watercolor"],
            "lighting": ["daylight", "sunset", "studio", "dramatic", "neon"],
            "camera": ["50mm", "portrait lens", "wide angle", "macro", "telephoto"],
            "elements": ["mountains", "forest", "river", "ocean", "city", "castle", "flowers"]
        }
    
    def create_or_update_elements_file(self):
        """要素ファイルが存在しない場合は作成する
        
        Returns:
            bool: 作成/更新成功時はTrue、失敗時はFalse
        """
        try:
            if not os.path.exists(self.elements_path):
                return self.save_elements(self.get_default_elements())
            return True
        except Exception as e:
            logger.error(f"要素ファイル作成エラー: {e}")
            return False 